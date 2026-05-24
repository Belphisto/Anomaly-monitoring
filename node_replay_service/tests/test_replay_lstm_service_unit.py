from pathlib import Path
import numpy as np
import pytest
import replay_lstm_service as service

def test_calibrate_threshold_uses_combined_score_distribution(sample_df):
    model=service.LSTMAutoencoderModel(window_size=3)
    threshold=service.calibrate_threshold(df=sample_df, model=model, calibration_fraction=1.0, threshold_percentile=0.5, threshold_margin=0.2)
    expected=float(np.quantile([12.5,14.0,15.5,17.0], 0.5) * 1.2)
    assert threshold == pytest.approx(expected)

def test_list_prepared_node_files_filters_hidden_and_respects_limit(tmp_path):
    (tmp_path/'node002.csv').write_text('x', encoding='utf-8')
    (tmp_path/'node001.csv').write_text('x', encoding='utf-8')
    (tmp_path/'_meta.csv').write_text('x', encoding='utf-8')
    files=service.list_prepared_node_files(tmp_path, max_nodes=2)
    assert [p.name for p in files] == ['node001.csv','node002.csv']

def test_step_node_updates_all_scores_and_flag(sample_df):
    model=service.LSTMAutoencoderModel(window_size=3)
    scorer=service.LSTMReplayScorer(model)
    state=service.NodeState(node='node001', df=sample_df.copy(), scorer=scorer, threshold=13.0)
    replay=service.ReplayService(prepared_dir=Path('.'), model=model, tick_seconds=0.1, rows_per_tick=4, calibration_fraction=0.5, threshold_percentile=0.99, threshold_margin=0.05)
    replay.step_node(state)
    assert state.position == 4
    assert state.last_score_cpu == pytest.approx(3.0)
    assert state.last_score_voltage == pytest.approx(11.0)
    assert state.last_score_combined == pytest.approx(14.0)
    assert state.last_flag == 1

def test_step_node_marks_completed_after_last_row(sample_df):
    model=service.LSTMAutoencoderModel(window_size=3)
    scorer=service.LSTMReplayScorer(model)
    state=service.NodeState(node='node001', df=sample_df.copy(), scorer=scorer, threshold=0.0)
    replay=service.ReplayService(prepared_dir=Path('.'), model=model, tick_seconds=0.1, rows_per_tick=10, calibration_fraction=0.5, threshold_percentile=0.99, threshold_margin=0.05)
    replay.step_node(state)
    assert state.completed is True
    assert state.position == len(sample_df)
