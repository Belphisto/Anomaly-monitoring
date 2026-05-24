from pathlib import Path

import numpy as np
import pytest
import replay_damp_service as service
from ._helpers import write_node_csv, record_sla_load_result

NODE_COUNTS = [12, 24, 36, 48, 96]
SERIES_LENGTH = 200
WINDOW_SIZE = 8
ANOMALY_START = 120
TOTAL_LOAD_SLA_SECONDS = 2.0
MAX_DETECTION_LATENCY_STEPS = 2


def _generate_damp_series(node_seed: int):
    rng = np.random.default_rng(10_000 + node_seed)
    x = np.linspace(0.0, 6.0 * np.pi, SERIES_LENGTH)
    cpu = 1.0 + 0.12 * np.sin(x) + rng.normal(0.0, 0.015, SERIES_LENGTH)
    voltage = 10.0 + 0.18 * np.cos(x / 2.0) + rng.normal(0.0, 0.02, SERIES_LENGTH)

    anomaly_positions = [ANOMALY_START, ANOMALY_START + 8, ANOMALY_START + 16]
    for offset, idx in enumerate(anomaly_positions):
        cpu[idx] += 8.0 + 0.5 * offset + rng.uniform(0.2, 0.4)
        voltage[idx] += 18.0 + 0.7 * offset + rng.uniform(0.3, 0.6)

    return cpu.round(6).tolist(), voltage.round(6).tolist()


def _build_service(prepared_dir: Path):
    replay = service.ReplayDAMPService(
        prepared_dir=prepared_dir,
        window_size=WINDOW_SIZE,
        start_index=32,
        tick_seconds=0.0,
        rows_per_tick=1,
        calibration_fraction=0.5,
        threshold_percentile=0.99,
        threshold_margin=0.0,
        init_backward_factor=8,
    )
    replay.initialize()
    return replay


@pytest.mark.parametrize("node_count", NODE_COUNTS, ids=lambda n: f"nodes_{n}")
def test_damp_load_sla_multi_node_replay(tmp_path, pytestconfig, node_count):
    prepared_dir = tmp_path / 'prepared'
    prepared_dir.mkdir()
    for i in range(node_count):
        cpu_values, voltage_values = _generate_damp_series(i)
        write_node_csv(prepared_dir / f'node{i:03d}.csv', cpu_values, voltage_values)

    replay = _build_service(prepared_dir)

    import time
    started = time.perf_counter()
    flag_positions = {}
    while not all(state.completed for state in replay.nodes.values()):
        for name, state in replay.nodes.items():
            if state.completed:
                continue
            replay.step_node(state)
            if int(state.last_flag) and state.position >= ANOMALY_START + 1 and name not in flag_positions:
                flag_positions[name] = int(state.position)
    elapsed = time.perf_counter() - started

    assert len(flag_positions) == node_count
    latencies = [pos - (ANOMALY_START + 1) for pos in flag_positions.values()]
    max_latency = max(latencies)
    avg_seconds = elapsed / node_count
    record_sla_load_result(
        pytestconfig,
        'DAMP',
        nodes=node_count,
        total_replay_seconds=elapsed,
        avg_replay_seconds=avg_seconds,
        max_latency_steps=max_latency,
    )
    assert elapsed < TOTAL_LOAD_SLA_SECONDS
    assert max_latency <= MAX_DETECTION_LATENCY_STEPS
