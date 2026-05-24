#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import logging
import os
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
LSTM_WORK_DIR = REPO_ROOT / "lstm_work"
if str(LSTM_WORK_DIR) not in sys.path:
    sys.path.insert(0, str(LSTM_WORK_DIR))
    
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from lstm_model import LSTMReplayScorer, LSTMAutoencoderModel, train_baseline_lstm


LOGGER = logging.getLogger("replay_lstm_service")
MODEL_ID = "lstm"
MODEL_LABEL_VALUE = "lstm"


# Legacy metrics
METRIC_CPU = Gauge(f"{MODEL_ID}_node_cpu_utilization", "Current replayed CPU utilization", ["node", "model"])
METRIC_VOLTAGE = Gauge(f"{MODEL_ID}_node_voltage_12v", "Current replayed +12V value", ["node", "model"])
METRIC_SCORE_CPU = Gauge(f"{MODEL_ID}_node_anomaly_score_cpu", "LSTM AE anomaly score for CPU", ["node", "model"])
METRIC_SCORE_VOLTAGE = Gauge(f"{MODEL_ID}_node_anomaly_score_voltage", "LSTM AE anomaly score for voltage", ["node", "model"])
METRIC_SCORE_COMBINED = Gauge(f"{MODEL_ID}_node_anomaly_score_combined", "LSTM AE combined anomaly score", ["node", "model"])
METRIC_THRESHOLD = Gauge(f"{MODEL_ID}_node_anomaly_threshold", "Per-node anomaly threshold", ["node", "model"])
METRIC_FLAG = Gauge(f"{MODEL_ID}_node_anomaly_flag", "Binary anomaly flag", ["node", "model"])
METRIC_LAST_TS = Gauge(f"{MODEL_ID}_node_last_timestamp_unix", "Last replayed timestamp in unix seconds", ["node", "model"])
METRIC_POSITION = Gauge(f"{MODEL_ID}_node_stream_position", "Current replay row position", ["node", "model"])
METRIC_WINDOW_READY = Gauge(f"{MODEL_ID}_node_window_ready", "1 when enough points are collected for LSTM window", ["node", "model"])
METRIC_REPLAY_ACTIVE = Gauge(f"{MODEL_ID}_replay_active", "1 when replay is active", ["model"])
METRIC_REPLAY_SPEED = Gauge(f"{MODEL_ID}_replay_rows_per_tick", "Replay rows per tick", ["model"])

# Unified metrics
COMMON_CPU = Gauge("anomaly_node_cpu_utilization", "Current replayed CPU utilization for any anomaly model", ["node", "model"])
COMMON_VOLTAGE = Gauge("anomaly_node_voltage_12v", "Current replayed +12V value for any anomaly model", ["node", "model"])
COMMON_SCORE = Gauge("anomaly_node_anomaly_score", "Primary anomaly score for any anomaly model", ["node", "model"])
COMMON_SCORE_BY_CHANNEL = Gauge("anomaly_node_anomaly_score_by_channel", "Per-channel anomaly score for any anomaly model", ["node", "model", "channel"])
COMMON_THRESHOLD = Gauge("anomaly_node_anomaly_threshold", "Per-node anomaly threshold for any anomaly model", ["node", "model"])
COMMON_FLAG = Gauge("anomaly_node_anomaly_flag", "Binary anomaly flag for any anomaly model", ["node", "model"])
COMMON_LAST_TS = Gauge("anomaly_node_last_timestamp_unix", "Last replayed timestamp for any anomaly model", ["node", "model"])
COMMON_POSITION = Gauge("anomaly_node_stream_position", "Replay row position for any anomaly model", ["node", "model"])
COMMON_WINDOW_READY = Gauge("anomaly_node_window_ready", "1 when enough points are collected for inference window", ["node", "model"])
COMMON_REPLAY_ACTIVE = Gauge("anomaly_replay_active", "1 when at least one node is still replaying in a model service", ["model"])
COMMON_REPLAY_SPEED = Gauge("anomaly_replay_rows_per_tick", "Configured replay speed for a model service", ["model"])
SERVICE_INFO = Info(f"{MODEL_ID}_service", "Information about LSTM replay service")
COMMON_SERVICE_INFO = Info("anomaly_service", "Unified information about anomaly replay service")


@dataclass
class NodeState:
    node: str
    df: pd.DataFrame
    scorer: LSTMReplayScorer
    threshold: float
    position: int = 0
    last_score_cpu: float = 0.0
    last_score_voltage: float = 0.0
    last_score_combined: float = 0.0
    last_flag: int = 0
    completed: bool = False


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
    model: LSTMAutoencoderModel,
    calibration_fraction: float,
    threshold_percentile: float,
    threshold_margin: float,
) -> float:
    calibr_rows = max(model.window_size + 1, int(len(df) * calibration_fraction))
    calibr_rows = min(calibr_rows, len(df))
    scorer = LSTMReplayScorer(model)
    scores: List[float] = []
    for _, row in df.iloc[:calibr_rows].iterrows():
        result = scorer.update(float(row["cpu"]), float(row["voltage"]))
        if result is not None and np.isfinite(result["combined"]):
            scores.append(float(result["combined"]))
    if not scores:
        return 1.0
    threshold = float(np.quantile(scores, threshold_percentile))
    threshold *= (1.0 + threshold_margin)
    return max(threshold, 1e-9)


def set_initial_metrics(node: str, threshold: float) -> None:
    labels = {"node": node, "model": MODEL_LABEL_VALUE}
    METRIC_SCORE_CPU.labels(**labels).set(0.0)
    METRIC_SCORE_VOLTAGE.labels(**labels).set(0.0)
    METRIC_SCORE_COMBINED.labels(**labels).set(0.0)
    METRIC_THRESHOLD.labels(**labels).set(float(threshold))
    METRIC_FLAG.labels(**labels).set(0)
    METRIC_POSITION.labels(**labels).set(0)
    METRIC_WINDOW_READY.labels(**labels).set(0)

    COMMON_SCORE.labels(**labels).set(0.0)
    COMMON_SCORE_BY_CHANNEL.labels(node=node, model=MODEL_LABEL_VALUE, channel="cpu").set(0.0)
    COMMON_SCORE_BY_CHANNEL.labels(node=node, model=MODEL_LABEL_VALUE, channel="voltage").set(0.0)
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
        cpu_value = float(row["cpu"])
        voltage_value = float(row["voltage"])
        ts_unix = float(pd.Timestamp(row["timestamp"]).timestamp())

        METRIC_CPU.labels(**labels).set(cpu_value)
        METRIC_VOLTAGE.labels(**labels).set(voltage_value)
        METRIC_LAST_TS.labels(**labels).set(ts_unix)

        COMMON_CPU.labels(**labels).set(cpu_value)
        COMMON_VOLTAGE.labels(**labels).set(voltage_value)
        COMMON_LAST_TS.labels(**labels).set(ts_unix)

    METRIC_SCORE_CPU.labels(**labels).set(state.last_score_cpu)
    METRIC_SCORE_VOLTAGE.labels(**labels).set(state.last_score_voltage)
    METRIC_SCORE_COMBINED.labels(**labels).set(state.last_score_combined)
    METRIC_THRESHOLD.labels(**labels).set(state.threshold)
    METRIC_FLAG.labels(**labels).set(state.last_flag)
    METRIC_POSITION.labels(**labels).set(state.position)
    METRIC_WINDOW_READY.labels(**labels).set(1 if state.scorer.ready else 0)

    COMMON_SCORE.labels(**labels).set(state.last_score_combined)
    COMMON_SCORE_BY_CHANNEL.labels(node=state.node, model=MODEL_LABEL_VALUE, channel="cpu").set(state.last_score_cpu)
    COMMON_SCORE_BY_CHANNEL.labels(node=state.node, model=MODEL_LABEL_VALUE, channel="voltage").set(state.last_score_voltage)
    COMMON_SCORE_BY_CHANNEL.labels(node=state.node, model=MODEL_LABEL_VALUE, channel="combined").set(state.last_score_combined)
    COMMON_THRESHOLD.labels(**labels).set(state.threshold)
    COMMON_FLAG.labels(**labels).set(state.last_flag)
    COMMON_POSITION.labels(**labels).set(state.position)
    COMMON_WINDOW_READY.labels(**labels).set(1 if state.scorer.ready else 0)


class ReplayService:
    def __init__(
        self,
        prepared_dir: Path,
        model: LSTMAutoencoderModel,
        tick_seconds: float,
        rows_per_tick: int,
        calibration_fraction: float,
        threshold_percentile: float,
        threshold_margin: float,
        max_nodes: Optional[int] = None,
        thresholds_out: Optional[Path] = None,
    ) -> None:
        self.prepared_dir = prepared_dir
        self.model = model
        self.tick_seconds = tick_seconds
        self.rows_per_tick = rows_per_tick
        self.calibration_fraction = calibration_fraction
        self.threshold_percentile = threshold_percentile
        self.threshold_margin = threshold_margin
        self.max_nodes = max_nodes
        self.thresholds_out = thresholds_out
        self.nodes: Dict[str, NodeState] = {}

    def initialize(self) -> None:
        node_files = list_prepared_node_files(self.prepared_dir, self.max_nodes)
        if not node_files:
            raise FileNotFoundError(f"No prepared node files found in {self.prepared_dir}")

        thresholds_payload = {}
        for path in node_files:
            node_name = path.stem
            df = load_prepared_node_df(path)
            if len(df) < self.model.window_size + 1:
                LOGGER.warning("Node %s skipped: too short (%s rows)", node_name, len(df))
                continue
            threshold = calibrate_threshold(
                df=df,
                model=self.model,
                calibration_fraction=self.calibration_fraction,
                threshold_percentile=self.threshold_percentile,
                threshold_margin=self.threshold_margin,
            )
            state = NodeState(node=node_name, df=df, scorer=LSTMReplayScorer(self.model), threshold=threshold)
            self.nodes[node_name] = state
            thresholds_payload[node_name] = {
                "threshold": threshold,
                "rows": int(len(df)),
                "calibration_fraction": self.calibration_fraction,
                "threshold_percentile": self.threshold_percentile,
                "threshold_margin": self.threshold_margin,
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
            result = state.scorer.update(float(row["cpu"]), float(row["voltage"]))
            if result is not None:
                state.last_score_cpu = float(result["cpu"])
                state.last_score_voltage = float(result["voltage"])
                state.last_score_combined = float(result["combined"])
                state.last_flag = 1 if state.last_score_combined >= state.threshold else 0
            update_metrics_from_state(state)
            state.position += 1
        if state.position >= len(state.df):
            state.completed = True

    def run(self) -> None:
        LOGGER.info("LSTM replay started for %s nodes", len(self.nodes))
        while True:
            any_active = False
            for state in self.nodes.values():
                if not state.completed:
                    self.step_node(state)
                    any_active = True
            METRIC_REPLAY_ACTIVE.labels(model=MODEL_LABEL_VALUE).set(1 if any_active else 0)
            COMMON_REPLAY_ACTIVE.labels(model=MODEL_LABEL_VALUE).set(1 if any_active else 0)
            if not any_active:
                LOGGER.info("LSTM replay finished for all nodes")
                time.sleep(max(1.0, self.tick_seconds))
                continue
            time.sleep(self.tick_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay microservice for baseline LSTM autoencoder + per-node threshold")
    parser.add_argument("--prepared_dir", type=str, required=True, help="Directory with prepared_nodes/*.csv")
    parser.add_argument("--lstm_output_dir", type=str, default=None, help="Directory with trained LSTM artifacts. Default: lstm_work/lstm_output")
    parser.add_argument("--baseline_node_name", type=str, default="node001", help="Node used for baseline training")
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--validation_split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--train_if_missing", action="store_true", help="Train baseline LSTM if artifacts are missing")
    parser.add_argument("--force_retrain", action="store_true", help="Retrain baseline LSTM even if artifacts already exist")
    parser.add_argument("--metrics_port", type=int, default=8002)
    parser.add_argument("--tick_seconds", type=float, default=0.2)
    parser.add_argument("--rows_per_tick", type=int, default=1)
    parser.add_argument("--calibration_fraction", type=float, default=0.10)
    parser.add_argument("--threshold_percentile", type=float, default=0.995)
    parser.add_argument("--threshold_margin", type=float, default=0.05)
    parser.add_argument("--max_nodes", type=int, default=None)
    parser.add_argument("--thresholds_out", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def ensure_model_artifacts(args: argparse.Namespace, prepared_dir: Path, output_dir: Path) -> None:
    model_path = output_dir / "lstm_autoencoder.pt"
    metadata_path = output_dir / "metadata.json"
    needs_train = args.force_retrain or not model_path.exists() or not metadata_path.exists()
    if not needs_train:
        return
    if not (args.train_if_missing or args.force_retrain):
        raise FileNotFoundError(
            f"PyTorch LSTM artifacts are missing in {output_dir}. Use --train_if_missing or --force_retrain."
        )
    LOGGER.info("Training baseline LSTM autoencoder on %s", args.baseline_node_name)
    train_baseline_lstm(
        prepared_dir=prepared_dir,
        output_dir=output_dir,
        baseline_node_name=args.baseline_node_name,
        window_size=args.window_size,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        patience=args.patience,
        verbose=1 if args.verbose else 0,
    )
    LOGGER.info("Baseline LSTM training finished")


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.verbose)

    prepared_dir = Path(args.prepared_dir).resolve()
    output_dir = Path(args.lstm_output_dir).resolve() if args.lstm_output_dir else (REPO_ROOT / "lstm_work" / "lstm_output")
    thresholds_out = Path(args.thresholds_out).resolve() if args.thresholds_out else None

    ensure_model_artifacts(args, prepared_dir, output_dir)
    model, artifacts = LSTMAutoencoderModel.load(output_dir)

    SERVICE_INFO.info({
        "model": MODEL_LABEL_VALUE,
        "training_node": artifacts.training_node,
        "window_size": str(artifacts.window_size),
        "latent_dim": str(artifacts.latent_dim),
        "prepared_dir": str(prepared_dir),
        "lstm_output_dir": str(output_dir),
        "framework": "pytorch",
        "device": getattr(artifacts, "device", "unknown"),
    })
    COMMON_SERVICE_INFO.info({
        "model": MODEL_LABEL_VALUE,
        "training_node": artifacts.training_node,
        "window_size": str(artifacts.window_size),
        "service": "replay_lstm_service",
        "framework": "pytorch",
        "device": getattr(artifacts, "device", "unknown"),
    })
    METRIC_REPLAY_SPEED.labels(model=MODEL_LABEL_VALUE).set(args.rows_per_tick)
    COMMON_REPLAY_SPEED.labels(model=MODEL_LABEL_VALUE).set(args.rows_per_tick)

    start_http_server(args.metrics_port)
    LOGGER.info("Prometheus metrics started on :%s/metrics", args.metrics_port)

    service = ReplayService(
        prepared_dir=prepared_dir,
        model=model,
        tick_seconds=args.tick_seconds,
        rows_per_tick=args.rows_per_tick,
        calibration_fraction=args.calibration_fraction,
        threshold_percentile=args.threshold_percentile,
        threshold_margin=args.threshold_margin,
        max_nodes=args.max_nodes,
        thresholds_out=thresholds_out,
    )
    service.initialize()
    service.run()


if __name__ == "__main__":
    main()
