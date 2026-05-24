#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

LOGGER = logging.getLogger("main_service")

@dataclass
class ModelProcessSpec:
    name: str
    script_path: Path
    port: int
    command: List[str]

SUPPORTED_MODEL_ALIASES = {
    "damp": "damp",
    "mdissid": "mdissid",
    "lstm": "lstm",
    "ae": "ae",
    "autoencoder": "ae",
}


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def normalize_models_arg(value: str) -> List[str]:
    raw = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not raw:
        raise ValueError("Empty --models value")
    if "all" in raw:
        return ["damp", "mdissid", "lstm", "ae"]
    result: List[str] = []
    for item in raw:
        if item not in SUPPORTED_MODEL_ALIASES:
            raise ValueError(f"Unsupported model: {item}")
        canonical = SUPPORTED_MODEL_ALIASES[item]
        if canonical not in result:
            result.append(canonical)
    return result


def add_common_replay_args(cmd: List[str], args: argparse.Namespace, metrics_port: int) -> None:
    cmd.extend([
        "--prepared_dir", str(Path(args.prepared_dir).resolve()),
        "--metrics_port", str(metrics_port),
        "--tick_seconds", str(args.tick_seconds),
        "--rows_per_tick", str(args.rows_per_tick),
    ])
    if args.max_nodes is not None:
        cmd.extend(["--max_nodes", str(args.max_nodes)])
    if args.verbose:
        cmd.append("--verbose")


def build_damp_spec(repo_root: Path, args: argparse.Namespace) -> ModelProcessSpec:
    script = repo_root / "node_replay_service" / "replay_damp_service.py"
    cmd = [sys.executable, str(script)]
    add_common_replay_args(cmd, args, args.damp_port)
    cmd.extend([
        "--window_size", str(args.damp_window_size),
        "--calibration_fraction", str(args.damp_calibration_fraction),
        "--threshold_percentile", str(args.damp_threshold_percentile),
        "--threshold_margin", str(args.damp_threshold_margin),
        "--init_backward_factor", str(args.damp_init_backward_factor),
    ])
    if args.damp_start_index is not None:
        cmd.extend(["--start_index", str(args.damp_start_index)])
    if args.thresholds_dir:
        cmd.extend(["--thresholds_out", str(Path(args.thresholds_dir).resolve() / "damp_thresholds.json")])
    return ModelProcessSpec("damp", script, args.damp_port, cmd)


def build_mdissid_spec(repo_root: Path, args: argparse.Namespace) -> ModelProcessSpec:
    missing = []
    for field_name in ["repo_snn_dir", "datasets_root", "results_root", "baseline_dataset_name"]:
        if not getattr(args, field_name):
            missing.append(f"--{field_name}")
    if missing:
        raise ValueError("Для запуска mDiSSiD нужно передать параметры: " + ", ".join(missing))
    script = repo_root / "node_replay_service" / "replay_mdissid_service.py"
    cmd = [sys.executable, str(script)]
    add_common_replay_args(cmd, args, args.mdissid_port)
    cmd.extend([
        "--repo_snn_dir", str(Path(args.repo_snn_dir).resolve()),
        "--datasets_root", str(Path(args.datasets_root).resolve()),
        "--results_root", str(Path(args.results_root).resolve()),
        "--baseline_dataset_name", str(args.baseline_dataset_name),
        "--nn_type", str(args.nn_type),
        "--calibration_fraction", str(args.mdissid_calibration_fraction),
        "--threshold_percentile", str(args.mdissid_threshold_percentile),
        "--threshold_margin", str(args.mdissid_threshold_margin),
    ])
    if args.thresholds_dir:
        cmd.extend(["--thresholds_out", str(Path(args.thresholds_dir).resolve() / "mdissid_thresholds.json")])
    return ModelProcessSpec("mdissid", script, args.mdissid_port, cmd)


def build_lstm_spec(repo_root: Path, args: argparse.Namespace) -> ModelProcessSpec:
    script = repo_root / "node_replay_service" / "replay_lstm_service.py"
    cmd = [sys.executable, str(script)]
    add_common_replay_args(cmd, args, args.lstm_port)
    cmd.extend([
        "--baseline_node_name", str(args.lstm_baseline_node_name),
        "--window_size", str(args.lstm_window_size),
        "--latent_dim", str(args.lstm_latent_dim),
        "--epochs", str(args.lstm_epochs),
        "--batch_size", str(args.lstm_batch_size),
        "--validation_split", str(args.lstm_validation_split),
        "--patience", str(args.lstm_patience),
        "--calibration_fraction", str(args.lstm_calibration_fraction),
        "--threshold_percentile", str(args.lstm_threshold_percentile),
        "--threshold_margin", str(args.lstm_threshold_margin),
    ])
    if args.lstm_output_dir:
        cmd.extend(["--lstm_output_dir", str(Path(args.lstm_output_dir).resolve())])
    if args.lstm_train_if_missing:
        cmd.append("--train_if_missing")
    if args.lstm_force_retrain:
        cmd.append("--force_retrain")
    if args.thresholds_dir:
        cmd.extend(["--thresholds_out", str(Path(args.thresholds_dir).resolve() / "lstm_thresholds.json")])
    return ModelProcessSpec("lstm", script, args.lstm_port, cmd)


def build_optional_spec(repo_root: Path, args: argparse.Namespace, model_name: str) -> Optional[ModelProcessSpec]:
    mapping = {
        "ae": (repo_root / "node_replay_service" / "replay_ae_service.py", args.ae_port),
    }
    script, port = mapping[model_name]
    if not script.exists():
        LOGGER.warning("Model %s skipped: script not found: %s", model_name, script)
        return None
    cmd = [sys.executable, str(script), "--prepared_dir", str(Path(args.prepared_dir).resolve()), "--metrics_port", str(port)]
    if args.max_nodes is not None:
        cmd.extend(["--max_nodes", str(args.max_nodes)])
    return ModelProcessSpec(model_name, script, port, cmd)


def build_specs(repo_root: Path, args: argparse.Namespace, models: List[str]) -> List[ModelProcessSpec]:
    specs: List[ModelProcessSpec] = []
    for model in models:
        if model == "damp":
            specs.append(build_damp_spec(repo_root, args))
        elif model == "mdissid":
            specs.append(build_mdissid_spec(repo_root, args))
        elif model == "lstm":
            specs.append(build_lstm_spec(repo_root, args))
        elif model == "ae":
            spec = build_optional_spec(repo_root, args, model)
            if spec is not None:
                specs.append(spec)
    return specs


def popen_specs(specs: List[ModelProcessSpec], workdir: Path) -> Dict[str, subprocess.Popen]:
    processes: Dict[str, subprocess.Popen] = {}

    for spec in specs:
        if not spec.script_path.exists():
            raise FileNotFoundError(f"Не найден файл сервиса модели: {spec.script_path}")

        env = os.environ.copy()

        # LSTM использует PyTorch. На старой GTX 650 CUDA недоступна для текущей версии PyTorch,
        # поэтому для LSTM-процесса принудительно отключаем CUDA.
        if spec.name == "lstm":
            env["CUDA_VISIBLE_DEVICES"] = "-1"
            env["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"

        LOGGER.info("Starting model=%s port=%s", spec.name, spec.port)
        LOGGER.info("Command: %s", shlex.join(spec.command))

        processes[spec.name] = subprocess.Popen(
            spec.command,
            cwd=str(workdir),
            env=env,
        )

    return processes


def terminate_processes(processes: Dict[str, subprocess.Popen], grace_seconds: float = 10.0) -> None:
    for name, proc in processes.items():
        if proc.poll() is None:
            LOGGER.info("Terminate: %s (pid=%s)", name, proc.pid)
            proc.terminate()
    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if all(proc.poll() is not None for proc in processes.values()):
            return
        time.sleep(0.2)
    for name, proc in processes.items():
        if proc.poll() is None:
            LOGGER.warning("Kill: %s (pid=%s)", name, proc.pid)
            proc.kill()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified launcher for replay-based anomaly models")
    parser.add_argument("--models", type=str, default="all")
    parser.add_argument("--prepared_dir", type=str, required=True)
    parser.add_argument("--max_nodes", type=int, default=25)
    parser.add_argument("--tick_seconds", type=float, default=0.2)
    parser.add_argument("--rows_per_tick", type=int, default=1)
    parser.add_argument("--thresholds_dir", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--damp_port", type=int, default=8001)
    parser.add_argument("--mdissid_port", type=int, default=8010)
    parser.add_argument("--lstm_port", type=int, default=8002)
    parser.add_argument("--ae_port", type=int, default=8003)

    parser.add_argument("--damp_window_size", type=int, default=20)
    parser.add_argument("--damp_start_index", type=int, default=None)
    parser.add_argument("--damp_calibration_fraction", type=float, default=0.10)
    parser.add_argument("--damp_threshold_percentile", type=float, default=0.995)
    parser.add_argument("--damp_threshold_margin", type=float, default=0.05)
    parser.add_argument("--damp_init_backward_factor", type=int, default=8)

    parser.add_argument("--repo_snn_dir", type=str, default=None)
    parser.add_argument("--datasets_root", type=str, default=None)
    parser.add_argument("--results_root", type=str, default=None)
    parser.add_argument("--baseline_dataset_name", type=str, default=None)
    parser.add_argument("--nn_type", type=str, default="FCN")
    parser.add_argument("--mdissid_calibration_fraction", type=float, default=0.10)
    parser.add_argument("--mdissid_threshold_percentile", type=float, default=0.995)
    parser.add_argument("--mdissid_threshold_margin", type=float, default=0.05)

    parser.add_argument("--lstm_output_dir", type=str, default=None)
    parser.add_argument("--lstm_baseline_node_name", type=str, default="node001")
    parser.add_argument("--lstm_window_size", type=int, default=20)
    parser.add_argument("--lstm_latent_dim", type=int, default=16)
    parser.add_argument("--lstm_epochs", type=int, default=30)
    parser.add_argument("--lstm_batch_size", type=int, default=64)
    parser.add_argument("--lstm_validation_split", type=float, default=0.1)
    parser.add_argument("--lstm_patience", type=int, default=5)
    parser.add_argument("--lstm_train_if_missing", action="store_true")
    parser.add_argument("--lstm_force_retrain", action="store_true")
    parser.add_argument("--lstm_calibration_fraction", type=float, default=0.10)
    parser.add_argument("--lstm_threshold_percentile", type=float, default=0.995)
    parser.add_argument("--lstm_threshold_margin", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.verbose)
    repo_root = Path(__file__).resolve().parent.parent
    models = normalize_models_arg(args.models)
    specs = build_specs(repo_root, args, models)
    if not specs:
        raise RuntimeError("No model services resolved")
    dup: Dict[int, List[str]] = {}
    for spec in specs:
        dup.setdefault(spec.port, []).append(spec.name)
    conflicts = {p: names for p, names in dup.items() if len(names) > 1}
    if conflicts:
        raise ValueError(f"Port conflicts detected: {conflicts}")

    processes: Dict[str, subprocess.Popen] = {}

    def _handler(signum, frame):
        LOGGER.info("Signal received: %s", signum)
        terminate_processes(processes)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    try:
        processes = popen_specs(specs, repo_root)
        while True:
            for name, proc in list(processes.items()):
                rc = proc.poll()
                if rc is not None:
                    raise RuntimeError(f"Model process '{name}' exited with code {rc}")
            time.sleep(1.0)
    finally:
        terminate_processes(processes)


if __name__ == "__main__":
    main()
