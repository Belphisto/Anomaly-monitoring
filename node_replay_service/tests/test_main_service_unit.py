from argparse import Namespace
import pytest
import main_service

def test_normalize_models_arg_with_aliases_and_deduplication():
    assert main_service.normalize_models_arg('damp,mdissid,autoencoder,lstm,damp') == ['damp','mdissid','ae','lstm']

def test_normalize_models_arg_all_keyword_expands_all_models():
    assert main_service.normalize_models_arg('all') == ['damp','mdissid','lstm','ae']

def test_normalize_models_arg_rejects_unknown_model():
    with pytest.raises(ValueError, match='Unsupported model'):
        main_service.normalize_models_arg('weird_model')

def test_add_common_replay_args_includes_expected_values(tmp_path):
    args=Namespace(prepared_dir=str(tmp_path/'prepared'), tick_seconds=0.2, rows_per_tick=3, max_nodes=7, verbose=True)
    cmd=['python','service.py']
    main_service.add_common_replay_args(cmd, args, metrics_port=8123)
    assert '--prepared_dir' in cmd and '--metrics_port' in cmd and '8123' in cmd
    assert '--tick_seconds' in cmd and '0.2' in cmd
    assert '--rows_per_tick' in cmd and '3' in cmd
    assert '--max_nodes' in cmd and '7' in cmd
    assert '--verbose' in cmd

def test_build_mdissid_spec_requires_mandatory_paths(tmp_path):
    args=Namespace(prepared_dir=str(tmp_path), tick_seconds=0.2, rows_per_tick=1, max_nodes=2, verbose=False, mdissid_port=8010, repo_snn_dir=None, datasets_root=None, results_root=None, baseline_dataset_name=None, nn_type='fcn', mdissid_calibration_fraction=0.1, mdissid_threshold_percentile=0.99, mdissid_threshold_margin=0.05, thresholds_dir=None)
    with pytest.raises(ValueError, match='mDiSSiD'):
        main_service.build_mdissid_spec(tmp_path, args)

def test_build_damp_spec_writes_thresholds_file_argument(tmp_path):
    args=Namespace(prepared_dir=str(tmp_path/'prepared'), tick_seconds=0.2, rows_per_tick=1, max_nodes=5, verbose=False, damp_port=8001, damp_window_size=20, damp_start_index=None, damp_calibration_fraction=0.1, damp_threshold_percentile=0.99, damp_threshold_margin=0.05, damp_init_backward_factor=8, thresholds_dir=str(tmp_path/'thresholds'))
    spec=main_service.build_damp_spec(tmp_path, args)
    assert spec.name=='damp' and spec.port==8001
    assert any('damp_thresholds.json' in part for part in spec.command)
