from __future__ import annotations

from pathlib import Path
import time
from typing import Callable

import pandas as pd

TOTAL_WALL_CLOCK_SLA_SECONDS = 0.25
MAX_DETECTION_LATENCY_STEPS = 2
LOAD_NODE_SIZES = [12, 24, 36, 48, 96]

def write_node_csv(path: Path, cpu_values, voltage_values):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=len(cpu_values), freq="30s"),
            "cpu": cpu_values,
            "voltage": voltage_values,
        }
    )
    df.to_csv(path, index=False)
    return df


def run_until_complete(state, step_fn: Callable[[object], None], score_getter: Callable[[object], float]):
    started = time.perf_counter()
    flags = []
    flag_positions = []
    positions = []
    scores = []
    first_flag_position = None
    while not state.completed:
        step_fn(state)
        current_position = int(state.position)
        positions.append(current_position)
        current_flag = int(state.last_flag)
        flags.append(current_flag)
        if current_flag:
            flag_positions.append(current_position)
        scores.append(float(score_getter(state)))
        if state.last_flag and first_flag_position is None:
            first_flag_position = current_position
    elapsed = time.perf_counter() - started
    return {
        "elapsed": elapsed,
        "flags": flags,
        "positions": positions,
        "flag_positions": flag_positions,
        "scores": scores,
        "first_flag_position": first_flag_position,
        "peak_score": max(scores) if scores else 0.0,
    }


def record_sla_result(pytestconfig, model: str, *, replay_seconds: float | None = None, latency_steps: int | None = None):
    store = getattr(pytestconfig, "_sla_results", None)
    if store is None:
        store = {}
        pytestconfig._sla_results = store
    model_row = store.setdefault(model, {})
    if replay_seconds is not None:
        model_row["replay_seconds"] = float(replay_seconds)
    if latency_steps is not None:
        model_row["latency_steps"] = int(latency_steps)



def record_sla_load_result(pytestconfig, model: str, *, nodes: int, total_replay_seconds: float, avg_replay_seconds: float, max_latency_steps: int):
    store = getattr(pytestconfig, "_sla_load_results", None)
    if store is None:
        store = {}
        pytestconfig._sla_load_results = store
    store[model] = {
        "nodes": int(nodes),
        "total_replay_seconds": float(total_replay_seconds),
        "avg_replay_seconds": float(avg_replay_seconds),
        "max_latency_steps": int(max_latency_steps),
    }
