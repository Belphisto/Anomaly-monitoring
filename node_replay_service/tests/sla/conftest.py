import sys, types
from pathlib import Path
import numpy as np
import pytest

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

class _ValueHolder:
    def __init__(self): self._v = 0.0
    def set(self, v): self._v = v
    def get(self): return self._v

class _MetricHandle:
    def __init__(self): self._value = _ValueHolder()
    def set(self, v): self._value.set(v)

class DummyGauge:
    def __init__(self, name, documentation, labelnames=None):
        self.name = name
        self.documentation = documentation
        self.labelnames = tuple(labelnames or [])
        self._children = {}
    def labels(self, **labels):
        key = tuple((label, labels.get(label)) for label in self.labelnames)
        self._children.setdefault(key, _MetricHandle())
        return self._children[key]

class DummyInfo(DummyGauge):
    def info(self, value): self._info = value

stub_prom = types.ModuleType('prometheus_client')
stub_prom.Gauge = DummyGauge
stub_prom.Info = DummyInfo
stub_prom.start_http_server = lambda *args, **kwargs: None
sys.modules['prometheus_client'] = stub_prom

class _StubDAMPResult:
    def __init__(self, score, position):
        self.score = score
        self.position = position

class StubMultidimDAMPStreamDetector:
    def __init__(self, window_size, start_index, init_backward_factor):
        self.window_size = window_size
        self.start_index = start_index
        self.init_backward_factor = init_backward_factor
        self.samples_seen = 0
    def update(self, cpu, voltage):
        self.samples_seen += 1
        if self.samples_seen < self.window_size:
            return None
        return _StubDAMPResult(float(cpu + voltage), self.samples_seen - self.window_size)

def stub_recommend_start_index(window_size): return window_size * 4

stub_damp = types.ModuleType('damp')
stub_damp.MultidimDAMPStreamDetector = StubMultidimDAMPStreamDetector
stub_damp.recommend_start_index = stub_recommend_start_index
sys.modules.setdefault('damp', stub_damp)

class StubLSTMReplayScorer:
    def __init__(self, model):
        self.model = model
        self.window = []
        self.ready = False
    def update(self, cpu, voltage):
        self.window.append((cpu, voltage))
        if len(self.window) < self.model.window_size:
            self.ready = False
            return None
        if len(self.window) > self.model.window_size:
            self.window.pop(0)
        self.ready = True
        cpu_vals = np.array([x[0] for x in self.window], dtype=float)
        volt_vals = np.array([x[1] for x in self.window], dtype=float)
        c = float(np.mean(cpu_vals))
        v = float(np.mean(volt_vals))
        return {'cpu': c, 'voltage': v, 'combined': c + v}

class StubLSTMAutoencoderModel:
    def __init__(self, window_size=3): self.window_size = window_size

stub_lstm = types.ModuleType('lstm_model')
stub_lstm.LSTMReplayScorer = StubLSTMReplayScorer
stub_lstm.LSTMAutoencoderModel = StubLSTMAutoencoderModel
stub_lstm.train_baseline_lstm = lambda *args, **kwargs: StubLSTMAutoencoderModel(window_size=kwargs.get('window_size', 3))
sys.modules.setdefault('lstm_model', stub_lstm)


def pytest_configure(config):
    config._sla_results = {}
    config._sla_load_results = {}


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    results = getattr(config, '_sla_results', {})
    if not results:
        return

    terminalreporter.write_sep('=', 'SLA summary')
    terminalreporter.write_line('| Модель | Время replay, с | Задержка, шаги |')
    terminalreporter.write_line('|---|---:|---:|')
    for model in ('DAMP', 'LSTM', 'mDISSID'):
        row = results.get(model, {})
        replay_seconds = row.get('replay_seconds')
        latency_steps = row.get('latency_steps')
        replay_cell = f'{replay_seconds:.5f}' if replay_seconds is not None else '-'
        latency_cell = str(latency_steps) if latency_steps is not None else '-'
        terminalreporter.write_line(f'| {model} | {replay_cell} | {latency_cell} |')

    load_results = getattr(config, '_sla_load_results', {})
    if load_results:
        terminalreporter.write_sep('=', 'SLA load summary')
        terminalreporter.write_line('| Модель | Узлов | Общее время replay, с | Среднее на узел, с | Макс. задержка, шаги |')
        terminalreporter.write_line('|---|---:|---:|---:|---:|')
        for model in ('DAMP', 'LSTM', 'mDISSID'):
            row = load_results.get(model, {})
            nodes = row.get('nodes')
            total_seconds = row.get('total_replay_seconds')
            avg_seconds = row.get('avg_replay_seconds')
            max_latency = row.get('max_latency_steps')
            nodes_cell = str(nodes) if nodes is not None else '-'
            total_cell = f'{total_seconds:.5f}' if total_seconds is not None else '-'
            avg_cell = f'{avg_seconds:.5f}' if avg_seconds is not None else '-'
            latency_cell = str(max_latency) if max_latency is not None else '-'
            terminalreporter.write_line(f'| {model} | {nodes_cell} | {total_cell} | {avg_cell} | {latency_cell} |')


@pytest.fixture
def sample_df():
    import pandas as pd
    return pd.DataFrame({
        'timestamp': pd.date_range('2026-01-01', periods=6, freq='30s'),
        'cpu': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'voltage': [10.0, 10.5, 11.0, 11.5, 12.0, 12.5],
    })
