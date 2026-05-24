from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import replay_mdissid_service as service
from ._helpers import (
    MAX_DETECTION_LATENCY_STEPS,
    TOTAL_WALL_CLOCK_SLA_SECONDS,
    run_until_complete,
    write_node_csv,
    record_sla_result,
)

SHIFT_START_INDEX = 6
WINDOW_SIZE = 3


class FakeEngine:
    def __init__(self, window_size: int):
        self.baseline_cfg = SimpleNamespace(window_size=window_size)

    def score_window(self, cpu_window, voltage_window):
        cpu_score = float(np.mean(cpu_window))
        voltage_score = float(np.mean(voltage_window))
        return cpu_score, voltage_score, cpu_score + voltage_score


def _build_service(prepared_dir: Path):
    replay = service.ReplayService(
        prepared_dir=prepared_dir,
        engine=FakeEngine(window_size=WINDOW_SIZE),
        tick_seconds=0.0,
        rows_per_tick=1,
        calibration_fraction=0.5,
        threshold_percentile=0.99,
        threshold_margin=0.0,
    )
    replay.initialize()
    return replay


def test_mdissid_sla_wall_clock_on_synthetic_stream(tmp_path, pytestconfig):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    write_node_csv(
        prepared_dir / "node001.csv",
        [1, 1, 1, 1, 1, 1, 9, 9, 9, 9],
        [10, 10, 10, 10, 10, 10, 20, 20, 20, 20],
    )
    replay = _build_service(prepared_dir)
    state = replay.nodes["node001"]

    result = run_until_complete(state, replay.step_node, lambda s: s.last_score_combined)

    record_sla_result(pytestconfig, 'mDISSID', replay_seconds=result['elapsed'])

    assert result["elapsed"] < TOTAL_WALL_CLOCK_SLA_SECONDS
    assert result["peak_score"] == pytest.approx(29.0)


def test_mdissid_sla_detects_shifted_window_with_limited_latency(tmp_path, pytestconfig):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    write_node_csv(
        prepared_dir / "node001.csv",
        [1, 1, 1, 1, 1, 1, 9, 9, 9, 9],
        [10, 10, 10, 10, 10, 10, 20, 20, 20, 20],
    )
    replay = _build_service(prepared_dir)
    state = replay.nodes["node001"]

    result = run_until_complete(state, replay.step_node, lambda s: s.last_score_combined)

    anomaly_observed_position = SHIFT_START_INDEX + 1
    flag_positions_after_onset = [pos for pos in result["flag_positions"] if pos >= anomaly_observed_position]

    assert flag_positions_after_onset
    detection_latency = flag_positions_after_onset[0] - anomaly_observed_position
    record_sla_result(pytestconfig, 'mDISSID', latency_steps=detection_latency)

    assert detection_latency <= MAX_DETECTION_LATENCY_STEPS
