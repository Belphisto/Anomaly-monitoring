#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Replay-микросервис для многозлового применения baseline-модели mDiSSiD.

Основная идея:
- одна baseline-модель, обученная на node001;
- для каждого узла калибруется свой порог аномальности;
- replay читает prepared_nodes/<node>.csv, подает точки "как поток";
- на каждом окне длины m считает anomaly score;
- публикует метрики в Prometheus.

Запуск из корня проекта, пример:
python .\node_replay_service\replay_mdissid_service.py ^
  --prepared_dir .\data\prepared_nodes ^
  --repo_snn_dir .\mdissid_work\mDiSSiD-main\src\SNN ^
  --datasets_root .\mdissid_work\mDiSSiD-main\datasets\SNN_datasets ^
  --results_root .\mdissid_work\mDiSSiD-main\SNN_results ^
  --baseline_dataset_name node001_multivariate_2641_20_10_2 ^
  --metrics_port 8010 ^
  --tick_seconds 0.2 ^
  --rows_per_tick 1

Prometheus target:
http://<host>:8010/metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from prometheus_client import Gauge, Info, start_http_server


LOGGER = logging.getLogger("replay_mdissid_service")
MODEL_ID = "mdissid"
MODEL_LABEL_VALUE = "mdissid"


# -----------------------------
# Метрики Prometheus
# -----------------------------

METRIC_CPU = Gauge(
    "mdissid_node_cpu_utilization",
    "Текущее воспроизводимое значение CPU utilization",
    ["node", "model"],
)
METRIC_VOLTAGE = Gauge(
    "mdissid_node_voltage_12v",
    "Текущее воспроизводимое значение напряжения +12V",
    ["node", "model"],
)
METRIC_SCORE_CPU = Gauge(
    "mdissid_node_anomaly_score_cpu",
    "Score baseline-модели по каналу CPU",
    ["node", "model"],
)
METRIC_SCORE_VOLTAGE = Gauge(
    "mdissid_node_anomaly_score_voltage",
    "Score baseline-модели по каналу voltage",
    ["node", "model"],
)
METRIC_SCORE_COMBINED = Gauge(
    "mdissid_node_anomaly_score_combined",
    "Итоговый score по узлу",
    ["node", "model"],
)
METRIC_THRESHOLD = Gauge(
    "mdissid_node_anomaly_threshold",
    "Индивидуальный порог аномальности узла",
    ["node", "model"],
)
METRIC_FLAG = Gauge(
    "mdissid_node_anomaly_flag",
    "Флаг аномалии: 1 - аномалия, 0 - норма",
    ["node", "model"],
)
METRIC_LAST_TS = Gauge(
    "mdissid_node_last_timestamp_unix",
    "Последняя временная метка replay в Unix time",
    ["node", "model"],
)
METRIC_POSITION = Gauge(
    "mdissid_node_stream_position",
    "Текущая позиция replay по узлу",
    ["node", "model"],
)
METRIC_WINDOW_READY = Gauge(
    "mdissid_node_window_ready",
    "Готовность окна инференса: 1 - окно заполнено, 0 - нет",
    ["node", "model"],
)
METRIC_REPLAY_ACTIVE = Gauge(
    "mdissid_replay_active",
    "Статус replay-сервиса: 1 - работает",
    ["model"],
)
METRIC_REPLAY_SPEED = Gauge(
    "mdissid_replay_rows_per_tick",
    "Количество строк, подаваемых за один tick",
    ["model"],
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
SERVICE_INFO = Info("mdissid_service", "Информация о replay-сервисе")
COMMON_SERVICE_INFO = Info("anomaly_service", "Unified information about anomaly replay service")


# -----------------------------
# Конфиг и модели состояния
# -----------------------------

@dataclass
class BaselineModelConfig:
    dataset_name: str
    nn_type: str
    window_size: int
    snippets_num_dim0: int
    snippets_num_dim1: int
    weight_path_dim0: Path
    weight_path_dim1: Path
    snippets_path_dim0: Path
    snippets_path_dim1: Path


@dataclass
class NodeState:
    node: str
    df: pd.DataFrame
    threshold: float
    position: int = 0
    cpu_buffer: Deque[float] = field(default_factory=deque)
    voltage_buffer: Deque[float] = field(default_factory=deque)
    last_score_cpu: float = 0.0
    last_score_voltage: float = 0.0
    last_score_combined: float = 0.0
    last_flag: int = 0
    completed: bool = False

    def init_buffers(self, window_size: int) -> None:
        self.cpu_buffer = deque(maxlen=window_size)
        self.voltage_buffer = deque(maxlen=window_size)


# -----------------------------
# Базовые утилиты
# -----------------------------

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _to_numeric_2d(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == object:
        rows = []
        for item in arr:
            if isinstance(item, (list, tuple, np.ndarray)):
                rows.append(np.asarray(item, dtype=float).reshape(-1))
            else:
                rows.append(np.asarray([item], dtype=float))
        arr = np.vstack(rows)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def normalize_dataset_like_training(X) -> np.ndarray:
    """
    Совместимо по смыслу с SNN/utils.py:
    MinMaxScaler по транспонированному представлению.
    """
    from sklearn.preprocessing import MinMaxScaler

    X = _to_numeric_2d(X)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X.T)
    return scaler.transform(X.T).T


def load_snippets(snippets_csv: Path, actual_snippets_num: int) -> np.ndarray:
    df = pd.read_csv(snippets_csv, header=None)
    x = df.iloc[:, :-1].to_numpy(dtype=float)
    x = normalize_dataset_like_training(x)
    x = x[:actual_snippets_num]
    return x


def make_ready_inputs_for_window(window: np.ndarray, snippets: np.ndarray) -> List[np.ndarray]:
    """
    Формирует список тензоров для mDiSSiD-модели:
    [win_vs_snippet0_left, win_vs_snippet0_right, win_vs_snippet1_left, ...]
    """
    ready_inputs: List[np.ndarray] = []
    window = np.asarray(window, dtype=float).reshape(1, -1, 1)

    for i in range(snippets.shape[0]):
        snippet = np.asarray(snippets[i], dtype=float).reshape(1, -1, 1)
        ready_inputs.append(window)
        ready_inputs.append(snippet)

    return ready_inputs


def load_snn_module(repo_snn_dir: Path):
    repo_snn_dir = repo_snn_dir.resolve()
    if str(repo_snn_dir) not in sys.path:
        sys.path.insert(0, str(repo_snn_dir))

    import multi_siamese_nn  # noqa
    return multi_siamese_nn


def load_baseline_config(
    datasets_root: Path,
    results_root: Path,
    baseline_dataset_name: str,
    nn_type: str,
) -> BaselineModelConfig:
    ds0 = datasets_root / baseline_dataset_name / "0"
    ds1 = datasets_root / baseline_dataset_name / "1"

    if not ds0.exists():
        raise FileNotFoundError(f"Не найден baseline dataset dir: {ds0}")
    if not ds1.exists():
        raise FileNotFoundError(f"Не найден baseline dataset dir: {ds1}")

    params0 = json.loads((ds0 / "input_params.json").read_text(encoding="utf-8"))
    params1 = json.loads((ds1 / "input_params.json").read_text(encoding="utf-8"))

    snn_params0_path = results_root / baseline_dataset_name / "0" / nn_type / "snn_params.json"
    snn_params1_path = results_root / baseline_dataset_name / "1" / nn_type / "snn_params.json"

    if not snn_params0_path.exists():
        raise FileNotFoundError(f"Не найден snn_params.json для dim0: {snn_params0_path}")
    if not snn_params1_path.exists():
        raise FileNotFoundError(f"Не найден snn_params.json для dim1: {snn_params1_path}")

    snn_params0 = json.loads(snn_params0_path.read_text(encoding="utf-8"))
    snn_params1 = json.loads(snn_params1_path.read_text(encoding="utf-8"))

    return BaselineModelConfig(
        dataset_name=baseline_dataset_name,
        nn_type=nn_type,
        window_size=int(params0["m"]),
        snippets_num_dim0=int(snn_params0.get("actual_snippets_num", params0["snippets_number"])),
        snippets_num_dim1=int(snn_params1.get("actual_snippets_num", params1["snippets_number"])),
        weight_path_dim0=results_root / baseline_dataset_name / "0" / nn_type / "models" / "weights.weights.h5",
        weight_path_dim1=results_root / baseline_dataset_name / "1" / nn_type / "models" / "weights.weights.h5",
        snippets_path_dim0=ds0 / "snippets.csv",
        snippets_path_dim1=ds1 / "snippets.csv",
    )


class BaselineInferenceEngine:
    def __init__(self, repo_snn_dir: Path, baseline_cfg: BaselineModelConfig):
        self.repo_snn_dir = repo_snn_dir
        self.baseline_cfg = baseline_cfg
        self.multi_siamese_nn = load_snn_module(repo_snn_dir)

        self.snippets_cpu = load_snippets(
            baseline_cfg.snippets_path_dim0,
            baseline_cfg.snippets_num_dim0,
        )
        self.snippets_voltage = load_snippets(
            baseline_cfg.snippets_path_dim1,
            baseline_cfg.snippets_num_dim1,
        )

        self.model_cpu = self._load_model(
            snippets_num=baseline_cfg.snippets_num_dim0,
            weight_path=baseline_cfg.weight_path_dim0,
        )
        self.model_voltage = self._load_model(
            snippets_num=baseline_cfg.snippets_num_dim1,
            weight_path=baseline_cfg.weight_path_dim1,
        )

    def _load_model(self, snippets_num: int, weight_path: Path):
        import tensorflow as tf  # local import to avoid eager loading before need

        if not weight_path.exists():
            raise FileNotFoundError(f"Не найден файл весов: {weight_path}")

        model = self.multi_siamese_nn.build_mDiSSiD_model(
            snippets_num=snippets_num,
            input_shape=(self.baseline_cfg.window_size, 1),
            base_type=self.baseline_cfg.nn_type,
        )
        tf.compat.v1.reset_default_graph()
        model.load_weights(str(weight_path))
        return model

    def score_window(self, cpu_window: np.ndarray, voltage_window: np.ndarray) -> Tuple[float, float, float]:
        cpu_window = normalize_dataset_like_training(np.asarray(cpu_window, dtype=float).reshape(1, -1))[0]
        voltage_window = normalize_dataset_like_training(np.asarray(voltage_window, dtype=float).reshape(1, -1))[0]

        ready_cpu = make_ready_inputs_for_window(cpu_window, self.snippets_cpu)
        ready_voltage = make_ready_inputs_for_window(voltage_window, self.snippets_voltage)

        pred_cpu = self.model_cpu.predict(ready_cpu, verbose=0)
        pred_voltage = self.model_voltage.predict(ready_voltage, verbose=0)

        score_cpu = float(np.min(np.asarray(pred_cpu).reshape(-1)))
        score_voltage = float(np.min(np.asarray(pred_voltage).reshape(-1)))
        combined_score = max(score_cpu, score_voltage)
        return score_cpu, score_voltage, combined_score


# -----------------------------
# Пороговая калибровка
# -----------------------------

def calibrate_node_threshold(
    node_df: pd.DataFrame,
    engine: BaselineInferenceEngine,
    window_size: int,
    calibration_fraction: float,
    threshold_percentile: float,
    threshold_margin: float,
) -> float:
    calibr_rows = max(window_size + 1, int(len(node_df) * calibration_fraction))
    calibr_rows = min(calibr_rows, len(node_df))

    cpu_values = node_df["cpu"].to_numpy(dtype=float)
    voltage_values = node_df["voltage"].to_numpy(dtype=float)

    scores: List[float] = []
    for end_idx in range(window_size, calibr_rows + 1):
        cpu_window = cpu_values[end_idx - window_size:end_idx]
        voltage_window = voltage_values[end_idx - window_size:end_idx]
        _, _, combined = engine.score_window(cpu_window, voltage_window)
        scores.append(combined)

    if not scores:
        return 1.0

    threshold = float(np.quantile(scores, threshold_percentile))
    threshold *= (1.0 + threshold_margin)
    return threshold


# -----------------------------
# Загрузка узлов и replay
# -----------------------------

def list_prepared_node_files(prepared_dir: Path, max_nodes: Optional[int] = None) -> List[Path]:
    files = []
    for p in sorted(prepared_dir.glob("*.csv")):
        if p.name.startswith("_"):
            continue
        files.append(p)

    if max_nodes is not None:
        files = files[:max_nodes]
    return files


def load_prepared_node_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "cpu", "voltage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} не содержит обязательные колонки: {missing}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["cpu"] = pd.to_numeric(df["cpu"], errors="coerce")
    df["voltage"] = pd.to_numeric(df["voltage"], errors="coerce")
    df = df.dropna(subset=["cpu", "voltage", "timestamp"]).reset_index(drop=True)
    return df


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


def update_metrics_from_state(node_state: NodeState) -> None:
    node = node_state.node
    labels = {"node": node, "model": MODEL_LABEL_VALUE}
    pos = min(node_state.position, len(node_state.df) - 1)
    if pos >= 0 and len(node_state.df) > 0:
        row = node_state.df.iloc[pos]
        cpu = float(row["cpu"])
        voltage = float(row["voltage"])
        ts = float(pd.Timestamp(row["timestamp"]).timestamp())
        METRIC_CPU.labels(**labels).set(cpu)
        METRIC_VOLTAGE.labels(**labels).set(voltage)
        METRIC_LAST_TS.labels(**labels).set(ts)
        COMMON_CPU.labels(**labels).set(cpu)
        COMMON_VOLTAGE.labels(**labels).set(voltage)
        COMMON_LAST_TS.labels(**labels).set(ts)

    METRIC_SCORE_CPU.labels(**labels).set(node_state.last_score_cpu)
    METRIC_SCORE_VOLTAGE.labels(**labels).set(node_state.last_score_voltage)
    METRIC_SCORE_COMBINED.labels(**labels).set(node_state.last_score_combined)
    METRIC_THRESHOLD.labels(**labels).set(node_state.threshold)
    METRIC_FLAG.labels(**labels).set(node_state.last_flag)
    METRIC_POSITION.labels(**labels).set(node_state.position)
    ready = 1 if len(node_state.cpu_buffer) == node_state.cpu_buffer.maxlen else 0
    METRIC_WINDOW_READY.labels(**labels).set(ready)

    COMMON_SCORE.labels(**labels).set(node_state.last_score_combined)
    COMMON_SCORE_BY_CHANNEL.labels(node=node, model=MODEL_LABEL_VALUE, channel="cpu").set(node_state.last_score_cpu)
    COMMON_SCORE_BY_CHANNEL.labels(node=node, model=MODEL_LABEL_VALUE, channel="voltage").set(node_state.last_score_voltage)
    COMMON_SCORE_BY_CHANNEL.labels(node=node, model=MODEL_LABEL_VALUE, channel="combined").set(node_state.last_score_combined)
    COMMON_THRESHOLD.labels(**labels).set(node_state.threshold)
    COMMON_FLAG.labels(**labels).set(node_state.last_flag)
    COMMON_POSITION.labels(**labels).set(node_state.position)
    COMMON_WINDOW_READY.labels(**labels).set(ready)


class ReplayService:
    def __init__(
        self,
        prepared_dir: Path,
        engine: BaselineInferenceEngine,
        tick_seconds: float,
        rows_per_tick: int,
        calibration_fraction: float,
        threshold_percentile: float,
        threshold_margin: float,
        max_nodes: Optional[int] = None,
        thresholds_out: Optional[Path] = None,
    ):
        self.prepared_dir = prepared_dir
        self.engine = engine
        self.tick_seconds = tick_seconds
        self.rows_per_tick = rows_per_tick
        self.calibration_fraction = calibration_fraction
        self.threshold_percentile = threshold_percentile
        self.threshold_margin = threshold_margin
        self.max_nodes = max_nodes
        self.thresholds_out = thresholds_out

        self.window_size = engine.baseline_cfg.window_size
        self.nodes: Dict[str, NodeState] = {}

    def initialize(self) -> None:
        node_files = list_prepared_node_files(self.prepared_dir, self.max_nodes)
        if not node_files:
            raise FileNotFoundError(f"В {self.prepared_dir} не найдено prepared node files")

        thresholds_payload = {}

        for path in node_files:
            node_name = path.stem
            df = load_prepared_node_df(path)
            if len(df) < self.window_size + 1:
                LOGGER.warning("Node %s skipped: too short (%s rows)", node_name, len(df))
                continue

            threshold = calibrate_node_threshold(
                node_df=df,
                engine=self.engine,
                window_size=self.window_size,
                calibration_fraction=self.calibration_fraction,
                threshold_percentile=self.threshold_percentile,
                threshold_margin=self.threshold_margin,
            )

            state = NodeState(node=node_name, df=df, threshold=threshold)
            state.init_buffers(self.window_size)
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
            self.thresholds_out.write_text(
                json.dumps(thresholds_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def step_node(self, state: NodeState) -> None:
        if state.completed:
            return

        for _ in range(self.rows_per_tick):
            if state.position >= len(state.df):
                state.completed = True
                break

            row = state.df.iloc[state.position]
            state.cpu_buffer.append(float(row["cpu"]))
            state.voltage_buffer.append(float(row["voltage"]))

            if len(state.cpu_buffer) == self.window_size:
                score_cpu, score_voltage, score_combined = self.engine.score_window(
                    np.asarray(state.cpu_buffer, dtype=float),
                    np.asarray(state.voltage_buffer, dtype=float),
                )
                state.last_score_cpu = score_cpu
                state.last_score_voltage = score_voltage
                state.last_score_combined = score_combined
                state.last_flag = 1 if score_combined >= state.threshold else 0

            update_metrics_from_state(state)
            state.position += 1

        if state.position >= len(state.df):
            state.completed = True

    def run(self) -> None:
        LOGGER.info("Replay started for %s nodes", len(self.nodes))
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
    parser = argparse.ArgumentParser(description="Replay-микросервис baseline mDiSSiD + per-node threshold")
    parser.add_argument("--prepared_dir", type=str, required=True, help="Папка с prepared_nodes/*.csv")
    parser.add_argument("--repo_snn_dir", type=str, required=True, help="Путь к mdissid_work/mDiSSiD-main/src/SNN")
    parser.add_argument("--datasets_root", type=str, required=True, help="Путь к datasets/SNN_datasets")
    parser.add_argument("--results_root", type=str, required=True, help="Путь к SNN_results")
    parser.add_argument("--baseline_dataset_name", type=str, required=True, help="Имя baseline dataset, например node001_multivariate_2641_20_10_2")
    parser.add_argument("--nn_type", type=str, default="FCN", help="Тип сети baseline-модели")
    parser.add_argument("--metrics_port", type=int, default=8010, help="Порт Prometheus /metrics")
    parser.add_argument("--tick_seconds", type=float, default=0.2, help="Пауза между replay ticks")
    parser.add_argument("--rows_per_tick", type=int, default=1, help="Сколько строк подавать за один tick")
    parser.add_argument("--calibration_fraction", type=float, default=0.1, help="Доля начальных точек узла для калибровки порога")
    parser.add_argument("--threshold_percentile", type=float, default=0.995, help="Перцентиль per-node threshold")
    parser.add_argument("--threshold_margin", type=float, default=0.05, help="Дополнительный запас к per-node threshold")
    parser.add_argument("--max_nodes", type=int, default=None, help="Ограничение числа узлов для replay")
    parser.add_argument("--thresholds_out", type=str, default=None, help="Куда сохранить JSON с per-node threshold")
    parser.add_argument("--verbose", action="store_true", help="Подробный лог")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.verbose)

    prepared_dir = Path(args.prepared_dir).resolve()
    repo_snn_dir = Path(args.repo_snn_dir).resolve()
    datasets_root = Path(args.datasets_root).resolve()
    results_root = Path(args.results_root).resolve()
    thresholds_out = Path(args.thresholds_out).resolve() if args.thresholds_out else None

    baseline_cfg = load_baseline_config(
        datasets_root=datasets_root,
        results_root=results_root,
        baseline_dataset_name=args.baseline_dataset_name,
        nn_type=args.nn_type,
    )

    SERVICE_INFO.info({
        "model": MODEL_ID,
        "baseline_dataset_name": args.baseline_dataset_name,
        "nn_type": args.nn_type,
        "window_size": str(baseline_cfg.window_size),
        "prepared_dir": str(prepared_dir),
    })
    COMMON_SERVICE_INFO.info({
        "model": MODEL_ID,
        "window_size": str(baseline_cfg.window_size),
        "prepared_dir": str(prepared_dir),
        "mode": "baseline_mdissid_2d",
    })
    METRIC_REPLAY_SPEED.labels(model=MODEL_LABEL_VALUE).set(args.rows_per_tick)
    COMMON_REPLAY_SPEED.labels(model=MODEL_LABEL_VALUE).set(args.rows_per_tick)

    start_http_server(args.metrics_port)
    LOGGER.info("Prometheus metrics started on :%s/metrics", args.metrics_port)

    engine = BaselineInferenceEngine(repo_snn_dir=repo_snn_dir, baseline_cfg=baseline_cfg)

    service = ReplayService(
        prepared_dir=prepared_dir,
        engine=engine,
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
