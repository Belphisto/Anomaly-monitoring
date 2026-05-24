from collections import deque
import numpy as np
import pandas as pd
import pytest
import replay_mdissid_service as service
class FakeEngine:
    def score_window(self, cpu_window, voltage_window):
        return float(np.mean(cpu_window)), float(np.mean(voltage_window)), float(np.mean(cpu_window)+np.mean(voltage_window))

def test_normalize_dataset_like_training_scales_rows_to_zero_one_range():
    x=np.array([[1.0,2.0,3.0],[10.0,20.0,30.0]])
    normalized=service.normalize_dataset_like_training(x)
    assert normalized.shape == x.shape
    assert np.all(normalized >= 0.0)
    assert np.all(normalized <= 1.0)

def test_make_ready_inputs_for_window_creates_pairs_for_each_snippet():
    ready=service.make_ready_inputs_for_window(np.array([1.0,2.0,3.0]), np.array([[0.1,0.2,0.3],[0.4,0.5,0.6]]))
    assert len(ready) == 4
    assert ready[0].shape == (1,3,1)
    assert ready[1].shape == (1,3,1)

def test_load_snippets_reads_all_rows_except_last_column(tmp_path):
    csv=tmp_path/'snippets.csv'
    pd.DataFrame([[1,2,3,99],[4,5,6,88]]).to_csv(csv, header=False, index=False)
    loaded=service.load_snippets(csv, actual_snippets_num=1)
    assert loaded.shape == (1,3)
    assert np.all(loaded >= 0.0)
    assert np.all(loaded <= 1.0)

def test_calibrate_node_threshold_uses_engine_scores(sample_df):
    threshold=service.calibrate_node_threshold(node_df=sample_df, engine=FakeEngine(), window_size=3, calibration_fraction=1.0, threshold_percentile=0.5, threshold_margin=0.1)
    expected=float(np.quantile([12.5,14.0,15.5,17.0], 0.5) * 1.1)
    assert threshold == pytest.approx(expected)

def test_node_state_init_buffers_creates_fixed_length_buffers():
    state=service.NodeState(node='n1', df=pd.DataFrame(), threshold=1.0)
    state.init_buffers(window_size=5)
    assert isinstance(state.cpu_buffer, deque)
    assert state.cpu_buffer.maxlen == 5
    assert state.voltage_buffer.maxlen == 5
