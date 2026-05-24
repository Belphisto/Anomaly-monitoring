from pathlib import Path

import pandas as pd
import pytest

import replay_damp_service as service


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


def test_initialize_pipeline_builds_nodes_and_thresholds_json(tmp_path):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    _write_node_csv(prepared_dir / "node001.csv", [1, 1, 1, 1, 1, 1], [10, 10, 10, 10, 10, 10])
    _write_node_csv(prepared_dir / "node002.csv", [2, 2, 2, 2, 2, 2], [11, 11, 11, 11, 11, 11])
    thresholds_out = tmp_path / "thresholds.json"

    replay = service.ReplayDAMPService(
        prepared_dir=prepared_dir,
        window_size=3,
        start_index=12,
        tick_seconds=0.0,
        rows_per_tick=1,
        calibration_fraction=0.5,
        threshold_percentile=0.9,
        threshold_margin=0.05,
        init_backward_factor=8,
        thresholds_out=thresholds_out,
    )
    replay.initialize()

    assert sorted(replay.nodes.keys()) == ["node001", "node002"]
    assert thresholds_out.exists()
    payload = pd.read_json(thresholds_out)
    assert set(payload.columns) == {"node001", "node002"}


def test_damp_pipeline_flags_synthetic_spike(tmp_path):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    _write_node_csv(
        prepared_dir / "node001.csv",
        [1, 1, 1, 1, 20, 1, 1],
        [10, 10, 10, 10, 50, 10, 10],
    )

    replay = service.ReplayDAMPService(
        prepared_dir=prepared_dir,
        window_size=3,
        start_index=12,
        tick_seconds=0.0,
        rows_per_tick=1,
        calibration_fraction=0.57,
        threshold_percentile=0.99,
        threshold_margin=0.0,
        init_backward_factor=8,
    )
    replay.initialize()
    state = replay.nodes["node001"]

    flags = []
    scores = []
    while not state.completed:
        replay.step_node(state)
        flags.append(state.last_flag)
        scores.append(state.last_score)

    assert max(scores) == pytest.approx(70.0)
    assert 1 in flags
