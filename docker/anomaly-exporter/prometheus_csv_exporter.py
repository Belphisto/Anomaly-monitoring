#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from wsgiref.simple_server import make_server


MONITORING_ROOT = Path("/monitoring_data")
TIMESERIES_FILE = MONITORING_ROOT / "timeseries" / "node001" / "dissid_points.csv"
ANOMALIES_FILE = MONITORING_ROOT / "anomalies" / "node001" / "dissid_regions.csv"
META_FILE = MONITORING_ROOT / "meta" / "dissid_node001_run.json"

MODEL_NAME = "dissid"
NODE_NAME = "node001"

# Сколько секунд держать одну точку перед переходом к следующей
# 1.0 = одна строка CSV в секунду
# 0.5 = две строки в секунду
# 2.0 = одна строка раз в 2 секунды
REPLAY_STEP_SECONDS = 0.02

# Если дошли до конца файла:
# True  -> начать заново с начала
# False -> зависнуть на последней точке
LOOP_REPLAY = True


class ReplayState:
    def __init__(self) -> None:
        self.df: Optional[pd.DataFrame] = None
        self.file_mtime: Optional[float] = None
        self.started_at: float = time.time()

    def _safe_float(self, v, default=0.0) -> float:
        try:
            if pd.isna(v):
                return default
            return float(v)
        except Exception:
            return default

    def _safe_int(self, v, default=0) -> int:
        try:
            if pd.isna(v):
                return default
            return int(v)
        except Exception:
            return default

    def _load_csv_if_needed(self) -> None:
        if not TIMESERIES_FILE.exists():
            self.df = None
            self.file_mtime = None
            return

        mtime = TIMESERIES_FILE.stat().st_mtime
        if self.df is not None and self.file_mtime == mtime:
            return

        df = pd.read_csv(TIMESERIES_FILE)
        df.columns = [str(c).strip() for c in df.columns]

        required_cols = [
            "timestamp",
            "cpu",
            "voltage",
            "relation_signal",
            "is_anomaly",
            "anomaly_region_id",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            self.df = None
            self.file_mtime = mtime
            return

        # Оставляем только нужные колонки и приводим типы
        df = df.copy()
        df["cpu"] = pd.to_numeric(df["cpu"], errors="coerce")
        df["voltage"] = pd.to_numeric(df["voltage"], errors="coerce")
        df["relation_signal"] = pd.to_numeric(df["relation_signal"], errors="coerce")
        df["is_anomaly"] = pd.to_numeric(df["is_anomaly"], errors="coerce").fillna(0).astype(int)
        df["anomaly_region_id"] = pd.to_numeric(df["anomaly_region_id"], errors="coerce").fillna(-1).astype(int)

        # timestamp читаем только для справки/отладки, но НЕ экспортируем
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        df = df.reset_index(drop=True)
        self.df = df
        self.file_mtime = mtime
        self.started_at = time.time()

    def get_current_row(self) -> tuple[Optional[pd.Series], Optional[int], Optional[int]]:
        self._load_csv_if_needed()

        if self.df is None or self.df.empty:
            return None, None, None

        elapsed = max(0.0, time.time() - self.started_at)
        raw_index = int(elapsed / REPLAY_STEP_SECONDS)

        if LOOP_REPLAY:
            index = raw_index % len(self.df)
        else:
            index = min(raw_index, len(self.df) - 1)

        return self.df.iloc[index], index, len(self.df)

    def build_metrics_text(self) -> str:
        lines: list[str] = []

        lines.append("# HELP anomaly_exporter_up Экспортер доступен")
        lines.append("# TYPE anomaly_exporter_up gauge")
        lines.append("anomaly_exporter_up 1")
        lines.append("")

        if not TIMESERIES_FILE.exists():
            lines.append("# HELP anomaly_results_available Есть ли файл результатов")
            lines.append("# TYPE anomaly_results_available gauge")
            lines.append(f'anomaly_results_available{{model="{MODEL_NAME}",node="{NODE_NAME}"}} 0')
            return "\n".join(lines) + "\n"

        lines.append("# HELP anomaly_results_available Есть ли файл результатов")
        lines.append("# TYPE anomaly_results_available gauge")
        lines.append(f'anomaly_results_available{{model="{MODEL_NAME}",node="{NODE_NAME}"}} 1')
        lines.append("")

        row, index, total = self.get_current_row()
        if row is None:
            lines.append("# HELP anomaly_results_schema_ok Корректна ли схема results CSV")
            lines.append("# TYPE anomaly_results_schema_ok gauge")
            lines.append(f'anomaly_results_schema_ok{{model="{MODEL_NAME}",node="{NODE_NAME}"}} 0')
            return "\n".join(lines) + "\n"

        lines.append("# HELP anomaly_results_schema_ok Корректна ли схема results CSV")
        lines.append("# TYPE anomaly_results_schema_ok gauge")
        lines.append(f'anomaly_results_schema_ok{{model="{MODEL_NAME}",node="{NODE_NAME}"}} 1')
        lines.append("")

        labels = f'model="{MODEL_NAME}",node="{NODE_NAME}"'

        cpu = self._safe_float(row["cpu"])
        voltage = self._safe_float(row["voltage"])
        relation_signal = self._safe_float(row["relation_signal"])
        is_anomaly = self._safe_int(row["is_anomaly"], 0)
        anomaly_region_id = self._safe_int(row["anomaly_region_id"], -1)

        lines.append("# HELP anomaly_input_cpu Текущее значение CPU из replay-потока")
        lines.append("# TYPE anomaly_input_cpu gauge")
        lines.append(f'anomaly_input_cpu{{{labels}}} {cpu}')
        lines.append("")

        lines.append("# HELP anomaly_input_voltage Текущее значение Voltage из replay-потока")
        lines.append("# TYPE anomaly_input_voltage gauge")
        lines.append(f'anomaly_input_voltage{{{labels}}} {voltage}')
        lines.append("")

        lines.append("# HELP anomaly_relation_signal Текущий relation_signal из replay-потока")
        lines.append("# TYPE anomaly_relation_signal gauge")
        lines.append(f'anomaly_relation_signal{{{labels}}} {relation_signal}')
        lines.append("")

        lines.append("# HELP anomaly_is_anomaly Текущий флаг аномалии")
        lines.append("# TYPE anomaly_is_anomaly gauge")
        lines.append(f'anomaly_is_anomaly{{{labels}}} {is_anomaly}')
        lines.append("")

        lines.append("# HELP anomaly_region_id Текущий идентификатор региона аномалии")
        lines.append("# TYPE anomaly_region_id gauge")
        lines.append(f'anomaly_region_id{{{labels}}} {anomaly_region_id}')
        lines.append("")

        lines.append("# HELP anomaly_replay_index Текущий индекс replay-потока")
        lines.append("# TYPE anomaly_replay_index gauge")
        lines.append(f'anomaly_replay_index{{{labels}}} {index}')
        lines.append("")

        lines.append("# HELP anomaly_replay_total_points Общее число точек в replay-файле")
        lines.append("# TYPE anomaly_replay_total_points gauge")
        lines.append(f'anomaly_replay_total_points{{{labels}}} {total}')
        lines.append("")

        lines.append("# HELP anomaly_replay_progress_ratio Прогресс прохождения по CSV от 0 до 1")
        lines.append("# TYPE anomaly_replay_progress_ratio gauge")
        ratio = 0.0 if not total else float(index) / float(max(1, total - 1))
        lines.append(f'anomaly_replay_progress_ratio{{{labels}}} {ratio}')
        lines.append("")

        return "\n".join(lines) + "\n"


STATE = ReplayState()


def app(environ, start_response):
    path = environ.get("PATH_INFO", "/")

    if path == "/health":
        body = b"ok"
        start_response(
            "200 OK",
            [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))],
        )
        return [body]

    if path == "/metrics":
        body = STATE.build_metrics_text().encode("utf-8")
        start_response(
            "200 OK",
            [
                ("Content-Type", "text/plain; version=0.0.4; charset=utf-8"),
                ("Content-Length", str(len(body))),
            ],
        )
        return [body]

    body = b"not found"
    start_response(
        "404 Not Found",
        [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))],
    )
    return [body]


if __name__ == "__main__":
    port = 8001
    print(f"Starting anomaly exporter on 0.0.0.0:{port}")
    print(f"Replay step seconds: {REPLAY_STEP_SECONDS}")
    print(f"Loop replay: {LOOP_REPLAY}")
    httpd = make_server("0.0.0.0", port, app)
    httpd.serve_forever()