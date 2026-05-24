from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import replay_damp_service as service

def test_list_prepared_node_files_filters_hidden_and_respects_limit(tmp_path):
    (tmp_path/'node002.csv').write_text('x', encoding='utf-8')
    (tmp_path/'node001.csv').write_text('x', encoding='utf-8')
    (tmp_path/'_meta.csv').write_text('x', encoding='utf-8')
    files=service.list_prepared_node_files(tmp_path, max_nodes=1)
    assert [p.name for p in files] == ['node001.csv']

def test_load_prepared_node_df_sorts_and_drops_invalid_rows(tmp_path):
    df=pd.DataFrame({'timestamp':['2026-01-01 00:01:00','2026-01-01 00:00:00','2026-01-01 00:02:00'],'cpu':[2,'bad',3],'voltage':[11.1,10.9,11.3]})
    path=tmp_path/'node001.csv'; df.to_csv(path, index=False)
    loaded=service.load_prepared_node_df(path)
    assert len(loaded)==2
    assert loaded.iloc[0]['timestamp'] < loaded.iloc[1]['timestamp']
    assert loaded['cpu'].tolist() == [2.0,3.0]

def test_calibrate_threshold_returns_quantile_with_margin(sample_df):
    threshold=service.calibrate_threshold(df=sample_df, window_size=3, start_index=12, calibration_fraction=1.0, threshold_percentile=0.5, threshold_margin=0.1, init_backward_factor=8)
    expected=float(np.quantile([14.0,15.5,17.0,18.5], 0.5) * 1.1)
    assert threshold == pytest.approx(expected)

def test_set_initial_metrics_populates_threshold_without_errors():
    service.set_initial_metrics('node001', 12.34)
    value=service.METRIC_THRESHOLD.labels(node='node001', model=service.MODEL_LABEL_VALUE)._value.get()
    assert value == pytest.approx(12.34)

def test_step_node_updates_score_flag_and_position(sample_df):
    detector=service.MultidimDAMPStreamDetector(window_size=3, start_index=12, init_backward_factor=8)
    state=service.NodeState(node='node001', df=sample_df.copy(), detector=detector, threshold=15.0)
    replay=service.ReplayDAMPService(prepared_dir=Path('.'), window_size=3, start_index=12, tick_seconds=0.1, rows_per_tick=4, calibration_fraction=0.5, threshold_percentile=0.99, threshold_margin=0.05, init_backward_factor=8)
    replay.step_node(state)
    assert state.position == 4
    assert state.last_score == pytest.approx(15.5)
    assert state.last_flag == 1
    assert state.subseq_index == 1

def test_step_node_marks_completed_when_dataframe_is_exhausted(sample_df):
    detector=service.MultidimDAMPStreamDetector(window_size=3, start_index=12, init_backward_factor=8)
    state=service.NodeState(node='node001', df=sample_df.copy(), detector=detector, threshold=100.0)
    replay=service.ReplayDAMPService(prepared_dir=Path('.'), window_size=3, start_index=12, tick_seconds=0.1, rows_per_tick=10, calibration_fraction=0.5, threshold_percentile=0.99, threshold_margin=0.05, init_backward_factor=8)
    replay.step_node(state)
    assert state.completed is True
    assert state.position == len(sample_df)
