from pathlib import Path

import pytest

import replay_damp_service as service
from ._helpers import (
    MAX_DETECTION_LATENCY_STEPS,
    TOTAL_WALL_CLOCK_SLA_SECONDS,
    run_until_complete,
    write_node_csv,
    record_sla_result,
)

SPIKE_INDEX = 6
WINDOW_SIZE = 3


def _build_service(prepared_dir: Path):
    replay = service.ReplayDAMPService(
        prepared_dir=prepared_dir,
        window_size=WINDOW_SIZE,
        start_index=12,
        tick_seconds=0.0,
        rows_per_tick=1,
        calibration_fraction=0.5,
        threshold_percentile=0.99,
        threshold_margin=0.0,
        init_backward_factor=8,
    )
    replay.initialize()
    return replay


def test_damp_sla_wall_clock_on_synthetic_stream(tmp_path, pytestconfig):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    write_node_csv(
        prepared_dir / "node001.csv",
        [1, 1, 1, 1, 1, 1, 20, 1, 1, 1],
        [10, 10, 10, 10, 10, 10, 50, 10, 10, 10],
    )
    replay = _build_service(prepared_dir)
    state = replay.nodes["node001"]

    result = run_until_complete(state, replay.step_node, lambda s: s.last_score)

    record_sla_result(pytestconfig, 'DAMP', replay_seconds=result['elapsed'])

    assert result["elapsed"] < TOTAL_WALL_CLOCK_SLA_SECONDS
    assert result["peak_score"] == pytest.approx(70.0)


def test_damp_sla_detects_spike_with_limited_latency(tmp_path, pytestconfig):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    write_node_csv(
        prepared_dir / "node001.csv",
        [1, 1, 1, 1, 1, 1, 20, 1, 1, 1],
        [10, 10, 10, 10, 10, 10, 50, 10, 10, 10],
    )
    replay = _build_service(prepared_dir)
    state = replay.nodes["node001"]

    result = run_until_complete(state, replay.step_node, lambda s: s.last_score)

    anomaly_observed_position = SPIKE_INDEX + 1
    flag_positions_after_onset = [pos for pos in result["flag_positions"] if pos >= anomaly_observed_position]

    assert flag_positions_after_onset
    detection_latency = flag_positions_after_onset[0] - anomaly_observed_position
    record_sla_result(pytestconfig, 'DAMP', latency_steps=detection_latency)

    assert detection_latency <= MAX_DETECTION_LATENCY_STEPS
