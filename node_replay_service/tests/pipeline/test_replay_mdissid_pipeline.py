from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import replay_mdissid_service as service


class FakeEngine:
    def __init__(self, window_size: int):
        self.baseline_cfg = SimpleNamespace(window_size=window_size)

    def score_window(self, cpu_window, voltage_window):
        cpu_score = float(np.mean(cpu_window))
        voltage_score = float(np.mean(voltage_window))
        return cpu_score, voltage_score, cpu_score + voltage_score


def _write_node_csv(path: Path, cpu_values, voltage_values):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=len(cpu_values), freq="30s"),
            "cpu": cpu_values,
            "voltage": voltage_values,
        }
    )
    df.to_csv(path, index=False)
    return df


def test_initialize_pipeline_builds_mdissid_nodes_and_buffers(tmp_path):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    _write_node_csv(prepared_dir / "node001.csv", [1, 2, 3, 4, 5], [10, 10, 10, 10, 10])
    _write_node_csv(prepared_dir / "node002.csv", [2, 3, 4, 5, 6], [11, 11, 11, 11, 11])
    thresholds_out = tmp_path / "thresholds.json"
    engine = FakeEngine(window_size=3)

    replay = service.ReplayService(
        prepared_dir=prepared_dir,
        engine=engine,
        tick_seconds=0.0,
        rows_per_tick=1,
        calibration_fraction=0.8,
        threshold_percentile=0.9,
        threshold_margin=0.05,
        thresholds_out=thresholds_out,
    )
    replay.initialize()

    assert sorted(replay.nodes.keys()) == ["node001", "node002"]
    assert all(node.cpu_buffer.maxlen == 3 for node in replay.nodes.values())
    assert thresholds_out.exists()


def test_mdissid_pipeline_flags_synthetic_shifted_window(tmp_path):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    _write_node_csv(
        prepared_dir / "node001.csv",
        [1, 1, 1, 1, 9, 9, 9],
        [10, 10, 10, 10, 20, 20, 20],
    )
    engine = FakeEngine(window_size=3)

    replay = service.ReplayService(
        prepared_dir=prepared_dir,
        engine=engine,
        tick_seconds=0.0,
        rows_per_tick=1,
        calibration_fraction=0.57,
        threshold_percentile=0.99,
        threshold_margin=0.0,
    )
    replay.initialize()
    state = replay.nodes["node001"]

    flags = []
    scores = []
    while not state.completed:
        replay.step_node(state)
        flags.append(state.last_flag)
        scores.append(state.last_score_combined)

    assert max(scores) == pytest.approx(29.0)
    assert 1 in flags
