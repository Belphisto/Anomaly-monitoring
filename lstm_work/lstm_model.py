#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple
from collections import deque

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


FEATURES = ["cpu", "voltage"]


def get_torch_device() -> torch.device:
    """Return CUDA device when it is available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def robust_center_scale(values: np.ndarray, eps: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    scale = 1.4826 * mad
    std = np.std(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > eps), scale, np.where(std > eps, std, 1.0))
    return median.astype(float), scale.astype(float)


def normalize_features(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (values - center) / scale


def denormalize_features(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return values * scale + center


def create_windows(values: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("Expected 2D array [n_samples, n_features]")
    n = len(values)
    if n < window_size:
        return np.empty((0, window_size, values.shape[1]), dtype=np.float32)
    windows = [values[i : i + window_size] for i in range(0, n - window_size + 1, stride)]
    return np.asarray(windows, dtype=np.float32)


@dataclass
class LSTMAEArtifacts:
    model_path: Path
    metadata_path: Path
    center: np.ndarray
    scale: np.ndarray
    window_size: int
    latent_dim: int
    training_node: str
    features: List[str]
    device: str


class _TorchLSTMAutoencoder(nn.Module):
    """PyTorch analogue of the original Keras LSTM autoencoder.

    Encoder: LSTM(2 -> 64) -> LSTM(64 -> 32) -> Linear(32 -> latent_dim) + ReLU
    Decoder: repeat latent vector -> LSTM(latent_dim -> 32) -> LSTM(32 -> 64) -> Linear(64 -> 2)
    """

    def __init__(self, window_size: int, n_features: int, latent_dim: int) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.n_features = int(n_features)
        self.latent_dim = int(latent_dim)

        self.encoder_lstm1 = nn.LSTM(
            input_size=self.n_features,
            hidden_size=64,
            batch_first=True,
        )
        self.encoder_lstm2 = nn.LSTM(
            input_size=64,
            hidden_size=32,
            batch_first=True,
        )
        self.to_latent = nn.Sequential(
            nn.Linear(32, self.latent_dim),
            nn.ReLU(),
        )
        self.decoder_lstm1 = nn.LSTM(
            input_size=self.latent_dim,
            hidden_size=32,
            batch_first=True,
        )
        self.decoder_lstm2 = nn.LSTM(
            input_size=32,
            hidden_size=64,
            batch_first=True,
        )
        self.output_layer = nn.Linear(64, self.n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.encoder_lstm1(x)
        _, (h_n, _) = self.encoder_lstm2(x)
        latent = self.to_latent(h_n[-1])
        repeated = latent.unsqueeze(1).repeat(1, self.window_size, 1)
        x, _ = self.decoder_lstm1(repeated)
        x, _ = self.decoder_lstm2(x)
        return self.output_layer(x)


class LSTMAutoencoderModel:
    def __init__(
        self,
        window_size: int = 20,
        n_features: int = 2,
        latent_dim: int = 16,
        device: Optional[str | torch.device] = None,
    ) -> None:
        self.window_size = int(window_size)
        self.n_features = int(n_features)
        self.latent_dim = int(latent_dim)
        self.device = torch.device(device) if device is not None else get_torch_device()
        self.model: Optional[_TorchLSTMAutoencoder] = None
        self.center: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None

    def build(self) -> _TorchLSTMAutoencoder:
        self.model = _TorchLSTMAutoencoder(
            window_size=self.window_size,
            n_features=self.n_features,
            latent_dim=self.latent_dim,
        ).to(self.device)
        return self.model

    def fit(
        self,
        train_values: np.ndarray,
        epochs: int = 30,
        batch_size: int = 64,
        validation_split: float = 0.1,
        patience: int = 5,
        verbose: int = 0,
    ) -> Dict[str, List[float]]:
        if self.model is None:
            self.build()
        assert self.model is not None

        center, scale = robust_center_scale(train_values)
        self.center = center.astype(np.float32)
        self.scale = scale.astype(np.float32)

        normalized = normalize_features(train_values, self.center, self.scale).astype(np.float32)
        windows = create_windows(normalized, self.window_size, stride=1)
        if len(windows) == 0:
            raise ValueError("Not enough points to build training windows")

        dataset = TensorDataset(torch.from_numpy(windows), torch.from_numpy(windows))
        val_size = int(len(dataset) * validation_split) if validation_split > 0 else 0
        val_size = min(max(val_size, 1), len(dataset) - 1) if len(dataset) > 1 and validation_split > 0 else 0
        train_size = len(dataset) - val_size
        if val_size > 0:
            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        else:
            train_dataset, val_dataset = dataset, None

        pin_memory = self.device.type == "cuda"
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=pin_memory,
        )
        val_loader = (
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=pin_memory)
            if val_dataset is not None
            else None
        )

        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_val = float("inf")
        epochs_without_improvement = 0
        history: Dict[str, List[float]] = {"loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_losses: List[float] = []
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                recon = self.model(xb)
                loss = criterion(recon, yb)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            history["loss"].append(train_loss)

            if val_loader is not None:
                val_loss = self._evaluate_loss(val_loader, criterion)
                history["val_loss"].append(val_loss)
                monitor = val_loss
            else:
                val_loss = float("nan")
                monitor = train_loss

            if verbose:
                if val_loader is not None:
                    print(f"Epoch {epoch:03d}/{epochs} - loss={train_loss:.6f} - val_loss={val_loss:.6f} - device={self.device}")
                else:
                    print(f"Epoch {epoch:03d}/{epochs} - loss={train_loss:.6f} - device={self.device}")

            if monitor < best_val - 1e-8:
                best_val = monitor
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if patience > 0 and epochs_without_improvement >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}; best loss={best_val:.6f}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        self.model.eval()
        return history

    @torch.no_grad()
    def _evaluate_loss(self, loader: DataLoader, criterion: nn.Module) -> float:
        assert self.model is not None
        self.model.eval()
        losses: List[float] = []
        for xb, yb in loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            losses.append(float(criterion(self.model(xb), yb).detach().cpu()))
        return float(np.mean(losses)) if losses else float("nan")

    def save(self, output_dir: Path, training_node: str) -> LSTMAEArtifacts:
        if self.model is None or self.center is None or self.scale is None:
            raise RuntimeError("Model is not trained")
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "lstm_autoencoder.pt"
        metadata_path = output_dir / "metadata.json"

        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "window_size": self.window_size,
                "n_features": self.n_features,
                "latent_dim": self.latent_dim,
                "features": FEATURES,
            },
            model_path,
        )
        metadata = {
            "framework": "pytorch",
            "model_file": model_path.name,
            "window_size": self.window_size,
            "latent_dim": self.latent_dim,
            "features": FEATURES,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "training_node": training_node,
            "device_saved_from": str(self.device),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return LSTMAEArtifacts(
            model_path=model_path,
            metadata_path=metadata_path,
            center=self.center.copy(),
            scale=self.scale.copy(),
            window_size=self.window_size,
            latent_dim=self.latent_dim,
            training_node=training_node,
            features=list(FEATURES),
            device=str(self.device),
        )

    @classmethod
    def load(cls, output_dir: Path, device: Optional[str | torch.device] = None) -> Tuple["LSTMAutoencoderModel", LSTMAEArtifacts]:
        metadata_path = output_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"LSTM metadata not found in {output_dir}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_file = metadata.get("model_file", "lstm_autoencoder.pt")
        model_path = output_dir / model_file
        if not model_path.exists():
            raise FileNotFoundError(f"PyTorch LSTM weights not found: {model_path}")

        selected_device = torch.device(device) if device is not None else get_torch_device()
        checkpoint = torch.load(model_path, map_location=selected_device)
        features = list(metadata.get("features", checkpoint.get("features", FEATURES)))
        model = cls(
            window_size=int(metadata["window_size"]),
            n_features=len(features),
            latent_dim=int(metadata.get("latent_dim", checkpoint.get("latent_dim", 16))),
            device=selected_device,
        )
        model.build()
        assert model.model is not None
        model.model.load_state_dict(checkpoint["state_dict"])
        model.model.to(selected_device)
        model.model.eval()
        model.center = np.asarray(metadata["center"], dtype=np.float32)
        model.scale = np.asarray(metadata["scale"], dtype=np.float32)

        artifacts = LSTMAEArtifacts(
            model_path=model_path,
            metadata_path=metadata_path,
            center=model.center.copy(),
            scale=model.scale.copy(),
            window_size=model.window_size,
            latent_dim=model.latent_dim,
            training_node=str(metadata.get("training_node", "node001")),
            features=features,
            device=str(selected_device),
        )
        return model, artifacts

    @torch.no_grad()
    def score_window(self, window_values: np.ndarray) -> Dict[str, float]:
        if self.model is None or self.center is None or self.scale is None:
            raise RuntimeError("Model is not loaded/trained")
        arr = np.asarray(window_values, dtype=np.float32)
        if arr.shape != (self.window_size, self.n_features):
            raise ValueError(f"Expected window shape {(self.window_size, self.n_features)}, got {arr.shape}")
        x = normalize_features(arr, self.center, self.scale).astype(np.float32)
        x_tensor = torch.from_numpy(x).unsqueeze(0).to(self.device, non_blocking=True)
        self.model.eval()
        recon = self.model(x_tensor).squeeze(0).detach().cpu().numpy()
        abs_err = np.abs(x - recon)
        cpu_score = float(np.mean(abs_err[:, 0]))
        voltage_score = float(np.mean(abs_err[:, 1]))
        combined_score = float(np.mean(abs_err))
        return {"cpu": cpu_score, "voltage": voltage_score, "combined": combined_score}


def load_prepared_node_values(path: Path, features: Sequence[str] = FEATURES) -> np.ndarray:
    df = pd.read_csv(path)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    values = df[list(features)].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=np.float32)
    return values


def train_baseline_lstm(
    prepared_dir: Path,
    output_dir: Path,
    baseline_node_name: str = "node001",
    window_size: int = 20,
    latent_dim: int = 16,
    epochs: int = 30,
    batch_size: int = 64,
    validation_split: float = 0.1,
    patience: int = 5,
    verbose: int = 0,
    device: Optional[str | torch.device] = None,
) -> LSTMAEArtifacts:
    baseline_path = prepared_dir / f"{baseline_node_name}.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline node file not found: {baseline_path}")

    train_values = load_prepared_node_values(baseline_path, FEATURES)
    model = LSTMAutoencoderModel(
        window_size=window_size,
        n_features=len(FEATURES),
        latent_dim=latent_dim,
        device=device,
    )
    model.fit(
        train_values=train_values,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        patience=patience,
        verbose=verbose,
    )
    return model.save(output_dir=output_dir, training_node=baseline_node_name)


class LSTMReplayScorer:
    def __init__(self, model: LSTMAutoencoderModel):
        self.model = model
        self.buffer: Deque[np.ndarray] = deque(maxlen=model.window_size)

    @property
    def ready(self) -> bool:
        return len(self.buffer) == self.buffer.maxlen

    def update(self, cpu_value: float, voltage_value: float) -> Optional[Dict[str, float]]:
        self.buffer.append(np.asarray([cpu_value, voltage_value], dtype=np.float32))
        if not self.ready:
            return None
        window = np.vstack(list(self.buffer))
        return self.model.score_window(window)
