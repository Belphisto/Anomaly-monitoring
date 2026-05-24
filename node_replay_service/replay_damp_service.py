#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Replay microservice for multidimensional DAMP on prepared node data.

Reads prepared_nodes/*.csv with columns:
- timestamp
- cpu
- voltage

Publishes Prometheus metrics so several models can coexist in one Grafana space.
Metric naming follows a common pattern:
- raw values:      <model>_node_cpu_utilization, <model>_node_voltage_12v
- scores/flags:    <model>_node_anomaly_score, <model>_node_anomaly_threshold, <model>_node_anomaly_flag
- service status:  <model>_node_stream_position, <model>_node_last_timestamp_unix, <model>_node_window_ready

Chosen model_id for this service: damp
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from prometheus_client import Gauge, Info, start_http_server

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
DAMP_WORK_DIR = REPO_ROOT / 'damp_work'
if str(DAMP_WORK_DIR) not in sys.path:
    sys.path.insert(0, str(DAMP_WORK_DIR))

from damp import MultidimDAMPStreamDetector, recommend_start_index


LOGGER = logging.getLogger("replay_damp_service")
MODEL_ID = "damp"
MODEL_LABEL_VALUE = "damp"


# -----------------------------
# Prometheus metrics
# -----------------------------

METRIC_CPU = Gauge(
    f"{MODEL_ID}_node_cpu_utilization",
    "Current replayed CPU utilization",
    ["node", "model"],
)
METRIC_VOLTAGE = Gauge(
    f"{MODEL_ID}_node_voltage_12v",
    "Current replayed +12V value",
    ["node", "model"],
)
METRIC_SCORE = Gauge(
    f"{MODEL_ID}_node_anomaly_score",
    "Multidimensional DAMP anomaly score for the current window",
    ["node", "model"],
)
METRIC_THRESHOLD = Gauge(
    f"{MODEL_ID}_node_anomaly_threshold",
    "Per-node anomaly threshold",
    ["node", "model"],
)
METRIC_FLAG = Gauge(
    f"{MODEL_ID}_node_anomaly_flag",
    "Binary anomaly flag: 1 anomaly, 0 normal",
    ["node", "model"],
)
METRIC_LAST_TS = Gauge(
    f"{MODEL_ID}_node_last_timestamp_unix",
    "Last replayed timestamp in unix seconds",
    ["node", "model"],
)
METRIC_POSITION = Gauge(
    f"{MODEL_ID}_node_stream_position",
    "Current replay row position for the node",
    ["node", "model"],
)
METRIC_WINDOW_READY = Gauge(
    f"{MODEL_ID}_node_window_ready",
    "1 when enough points are collected for a DAMP window",
    ["node", "model"],
)
METRIC_REPLAY_ACTIVE = Gauge(
    f"{MODEL_ID}_replay_active",
    "1 when at least one node is still replaying",
    ["model"],
)
METRIC_REPLAY_SPEED = Gauge(
    f"{MODEL_ID}_replay_rows_per_tick",
    "Configured replay speed in rows per tick",
    ["model"],
)
METRIC_SUBSEQ_INDEX = Gauge(
    f"{MODEL_ID}_node_subsequence_index",
    "Current subsequence start index processed by DAMP",
    ["node", "model"],
)

# Общие унифицированные метрики для Grafana/Prometheus
COMMON_CPU = Gauge(
    "anomaly_node_cpu_utilization",
    "Current replayed CPU utilization for any anomaly model",
    ["node", "model"],
)
COMMON_VOLTAGE = Gauge(
    "anomaly_node_voltage_12v",
    "Current replayed +12V value for any anomaly model",
    ["node", "model"],
)
COMMON_SCORE = Gauge(
    "anomaly_node_anomaly_score",
    "Primary anomaly score for any anomaly model",
    ["node", "model"],
)
COMMON_SCORE_BY_CHANNEL = Gauge(
    "anomaly_node_anomaly_score_by_channel",
    "Per-channel anomaly score for any anomaly model",
    ["node", "model", "channel"],
)
COMMON_THRESHOLD = Gauge(
    "anomaly_node_anomaly_threshold",
    "Per-node anomaly threshold for any anomaly model",
    ["node", "model"],
)
COMMON_FLAG = Gauge(
    "anomaly_node_anomaly_flag",
    "Binary anomaly flag for any anomaly model",
    ["node", "model"],
)
COMMON_LAST_TS = Gauge(
    "anomaly_node_last_timestamp_unix",
    "Last replayed timestamp for any anomaly model",
    ["node", "model"],
)
COMMON_POSITION = Gauge(
    "anomaly_node_stream_position",
    "Replay row position for any anomaly model",
    ["node", "model"],
)
COMMON_WINDOW_READY = Gauge(
    "anomaly_node_window_ready",
    "1 when enough points are collected for inference window",
    ["node", "model"],
)
COMMON_REPLAY_ACTIVE = Gauge(
    "anomaly_replay_active",
    "1 when at least one node is still replaying in a model service",
    ["model"],
)
COMMON_REPLAY_SPEED = Gauge(
    "anomaly_replay_rows_per_tick",
    "Configured replay speed for a model service",
    ["model"],
)
SERVICE_INFO = Info(f"{MODEL_ID}_service", "Information about DAMP replay service")
COMMON_SERVICE_INFO = Info("anomaly_service", "Unified information about anomaly replay service")


# -----------------------------
# Config / state
# -----------------------------

@dataclass
class NodeState:
    node: str
    df: pd.DataFrame
    detector: MultidimDAMPStreamDetector
    threshold: float
    position: int = 0
    last_score: float = 0.0
    last_flag: int = 0
    completed: bool = False
    subseq_index: int = -1


# -----------------------------
# Helpers
# -----------------------------

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def list_prepared_node_files(prepared_dir: Path, max_nodes: Optional[int] = None) -> List[Path]:
    files = [p for p in sorted(prepared_dir.glob("*.csv")) if not p.name.startswith("_")]
    if max_nodes is not None:
        files = files[:max_nodes]
    return files


def load_prepared_node_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "cpu", "voltage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} does not contain required columns: {sorted(missing)}")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["cpu"] = pd.to_numeric(df["cpu"], errors="coerce")
    df["voltage"] = pd.to_numeric(df["voltage"], errors="coerce")
    df = df.dropna(subset=["timestamp", "cpu", "voltage"]).reset_index(drop=True)
    return df


def calibrate_threshold(
    df: pd.DataFrame,
    window_size: int,
    start_index: int,
    calibration_fraction: float,
    threshold_percentile: float,
    threshold_margin: float,
    init_backward_factor: int,
) -> float:
    calibr_rows = max(window_size + 1, int(len(df) * calibration_fraction))
    calibr_rows = min(calibr_rows, len(df))

    detector = MultidimDAMPStreamDetector(
        window_size=window_size,
        start_index=start_index,
        init_backward_factor=init_backward_factor,
    )

    scores: List[float] = []
    for _, row in df.iloc[:calibr_rows].iterrows():
        result = detector.update(float(row["cpu"]), float(row["voltage"]))
        if result is not None and np.isfinite(result.score):
            scores.append(float(result.score))

    if not scores:
        return 1.0

    threshold = float(np.quantile(scores, threshold_percentile))
    threshold *= (1.0 + threshold_margin)
    return max(threshold, 1e-9)


def set_initial_metrics(node: str, threshold: float) -> None:
    labels = {"node": node, "model": MODEL_LABEL_VALUE}
    METRIC_SCORE.labels(**labels).set(0.0)
    METRIC_THRESHOLD.labels(**labels).set(float(threshold))
    METRIC_FLAG.labels(**labels).set(0)
    METRIC_POSITION.labels(**labels).set(0)
    METRIC_WINDOW_READY.labels(**labels).set(0)
    METRIC_SUBSEQ_INDEX.labels(**labels).set(-1)

    COMMON_SCORE.labels(**labels).set(0.0)
    COMMON_SCORE_BY_CHANNEL.labels(node=node, model=MODEL_LABEL_VALUE, channel="combined").set(0.0)
    COMMON_THRESHOLD.labels(**labels).set(float(threshold))
    COMMON_FLAG.labels(**labels).set(0)
    COMMON_POSITION.labels(**labels).set(0)
    COMMON_WINDOW_READY.labels(**labels).set(0)


def update_metrics_from_state(state: NodeState) -> None:
    labels = {"node": state.node, "model": MODEL_LABEL_VALUE}
    pos = min(state.position, len(state.df) - 1)
    if pos >= 0 and len(state.df) > 0:
        row = state.df.iloc[pos]
        cpu = float(row["cpu"])
        voltage = float(row["voltage"])
        ts = float(pd.Timestamp(row["timestamp"]).timestamp())
        METRIC_CPU.labels(**labels).set(cpu)
        METRIC_VOLTAGE.labels(**labels).set(voltage)
        METRIC_LAST_TS.labels(**labels).set(ts)
        COMMON_CPU.labels(**labels).set(cpu)
        COMMON_VOLTAGE.labels(**labels).set(voltage)
        COMMON_LAST_TS.labels(**labels).set(ts)

    METRIC_SCORE.labels(**labels).set(float(state.last_score))
    METRIC_THRESHOLD.labels(**labels).set(float(state.threshold))
    METRIC_FLAG.labels(**labels).set(int(state.last_flag))
    METRIC_POSITION.labels(**labels).set(int(state.position))
    ready = 1 if state.detector.samples_seen >= state.detector.window_size else 0
    METRIC_WINDOW_READY.labels(**labels).set(ready)
    METRIC_SUBSEQ_INDEX.labels(**labels).set(int(state.subseq_index))

    COMMON_SCORE.labels(**labels).set(float(state.last_score))
    COMMON_SCORE_BY_CHANNEL.labels(node=state.node, model=MODEL_LABEL_VALUE, channel="combined").set(float(state.last_score))
    COMMON_THRESHOLD.labels(**labels).set(float(state.threshold))
    COMMON_FLAG.labels(**labels).set(int(state.last_flag))
    COMMON_POSITION.labels(**labels).set(int(state.position))
    COMMON_WINDOW_READY.labels(**labels).set(ready)


# -----------------------------
# Service
# -----------------------------

class ReplayDAMPService:
    def __init__(
        self,
        prepared_dir: Path,
        window_size: int,
        start_index: Optional[int],
        tick_seconds: float,
        rows_per_tick: int,
        calibration_fraction: float,
        threshold_percentile: float,
        threshold_margin: float,
        init_backward_factor: int,
        max_nodes: Optional[int] = None,
        thresholds_out: Optional[Path] = None,
    ) -> None:
        self.prepared_dir = prepared_dir
        self.window_size = window_size
        self.start_index = recommend_start_index(window_size) if start_index is None else int(start_index)
        self.tick_seconds = tick_seconds
        self.rows_per_tick = rows_per_tick
        self.calibration_fraction = calibration_fraction
        self.threshold_percentile = threshold_percentile
        self.threshold_margin = threshold_margin
        self.init_backward_factor = init_backward_factor
        self.max_nodes = max_nodes
        self.thresholds_out = thresholds_out
        self.nodes: Dict[str, NodeState] = {}

    def initialize(self) -> None:
        node_files = list_prepared_node_files(self.prepared_dir, self.max_nodes)
        if not node_files:
            raise FileNotFoundError(f"No prepared node files found in {self.prepared_dir}")

        thresholds_payload: Dict[str, dict] = {}

        for path in node_files:
            node_name = path.stem
            df = load_prepared_node_df(path)
            if len(df) < self.window_size + 1:
                LOGGER.warning("Node %s skipped: too short (%s rows)", node_name, len(df))
                continue

            threshold = calibrate_threshold(
                df=df,
                window_size=self.window_size,
                start_index=self.start_index,
                calibration_fraction=self.calibration_fraction,
                threshold_percentile=self.threshold_percentile,
                threshold_margin=self.threshold_margin,
                init_backward_factor=self.init_backward_factor,
            )

            detector = MultidimDAMPStreamDetector(
                window_size=self.window_size,
                start_index=self.start_index,
                init_backward_factor=self.init_backward_factor,
            )
            state = NodeState(node=node_name, df=df, detector=detector, threshold=threshold)
            self.nodes[node_name] = state
            thresholds_payload[node_name] = {
                "threshold": float(threshold),
                "rows": int(len(df)),
                "window_size": int(self.window_size),
                "start_index": int(self.start_index),
                "calibration_fraction": float(self.calibration_fraction),
                "threshold_percentile": float(self.threshold_percentile),
                "threshold_margin": float(self.threshold_margin),
            }
            set_initial_metrics(node_name, threshold)
            LOGGER.info("Node %s ready: rows=%s threshold=%.6f", node_name, len(df), threshold)

        if self.thresholds_out:
            self.thresholds_out.parent.mkdir(parents=True, exist_ok=True)
            self.thresholds_out.write_text(json.dumps(thresholds_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def step_node(self, state: NodeState) -> None:
        if state.completed:
            return

        for _ in range(self.rows_per_tick):
            if state.position >= len(state.df):
                state.completed = True
                break

            row = state.df.iloc[state.position]
            result = state.detector.update(float(row["cpu"]), float(row["voltage"]))
            if result is not None and np.isfinite(result.score):
                state.last_score = float(result.score)
                state.last_flag = 1 if state.last_score >= state.threshold else 0
                state.subseq_index = int(result.position)

            update_metrics_from_state(state)
            state.position += 1

        if state.position >= len(state.df):
            state.completed = True

    def run(self) -> None:
        LOGGER.info("DAMP replay started for %s nodes", len(self.nodes))
        while True:
            any_active = False
            for state in self.nodes.values():
                if not state.completed:
                    self.step_node(state)
                    any_active = True

            METRIC_REPLAY_ACTIVE.labels(model=MODEL_LABEL_VALUE).set(1 if any_active else 0)
            COMMON_REPLAY_ACTIVE.labels(model=MODEL_LABEL_VALUE).set(1 if any_active else 0)
            if not any_active:
                LOGGER.info("Replay finished for all nodes")
                time.sleep(max(1.0, self.tick_seconds))
                continue

            time.sleep(self.tick_seconds)


# -----------------------------
# CLI
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay microservice for multidimensional DAMP on prepared nodes")
    parser.add_argument("--prepared_dir", type=str, required=True, help="Directory with prepared_nodes/*.csv")
    parser.add_argument("--window_size", type=int, default=20, help="DAMP subsequence length m")
    parser.add_argument("--start_index", type=int, default=None, help="Index of processing start in subsequence space; default is 4*m")
    parser.add_argument("--metrics_port", type=int, default=8001, help="Prometheus port for /metrics")
    parser.add_argument("--tick_seconds", type=float, default=0.2, help="Pause between replay ticks")
    parser.add_argument("--rows_per_tick", type=int, default=1, help="How many rows to replay per tick")
    parser.add_argument("--calibration_fraction", type=float, default=0.10, help="Initial node fraction used to calibrate threshold")
    parser.add_argument("--threshold_percentile", type=float, default=0.995, help="Percentile for per-node threshold")
    parser.add_argument("--threshold_margin", type=float, default=0.05, help="Extra margin added to threshold")
    parser.add_argument("--init_backward_factor", type=int, default=8, help="Initial backward search span multiplier before power-of-two rounding")
    parser.add_argument("--max_nodes", type=int, default=None, help="Limit number of replayed nodes")
    parser.add_argument("--thresholds_out", type=str, default=None, help="Where to save JSON with per-node thresholds")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.verbose)

    prepared_dir = Path(args.prepared_dir).resolve()
    thresholds_out = Path(args.thresholds_out).resolve() if args.thresholds_out else None

    SERVICE_INFO.info(
        {
            "model": MODEL_ID,
            "window_size": str(args.window_size),
            "start_index": str(args.start_index if args.start_index is not None else recommend_start_index(args.window_size)),
            "prepared_dir": str(prepared_dir),
            "mode": "streaming_left_damp_2d",
        }
    )
    COMMON_SERVICE_INFO.info(
        {
            "model": MODEL_ID,
            "window_size": str(args.window_size),
            "start_index": str(args.start_index if args.start_index is not None else recommend_start_index(args.window_size)),
            "prepared_dir": str(prepared_dir),
            "mode": "streaming_left_damp_2d",
        }
    )
    METRIC_REPLAY_SPEED.labels(model=MODEL_LABEL_VALUE).set(args.rows_per_tick)
    COMMON_REPLAY_SPEED.labels(model=MODEL_LABEL_VALUE).set(args.rows_per_tick)

    start_http_server(args.metrics_port)
    LOGGER.info("Prometheus metrics started on :%s/metrics", args.metrics_port)

    service = ReplayDAMPService(
        prepared_dir=prepared_dir,
        window_size=args.window_size,
        start_index=args.start_index,
        tick_seconds=args.tick_seconds,
        rows_per_tick=args.rows_per_tick,
        calibration_fraction=args.calibration_fraction,
        threshold_percentile=args.threshold_percentile,
        threshold_margin=args.threshold_margin,
        init_backward_factor=args.init_backward_factor,
        max_nodes=args.max_nodes,
        thresholds_out=thresholds_out,
    )
    service.initialize()
    service.run()


if __name__ == "__main__":
    main()
