#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multidimensional DAMP utilities for 2D node time series.

Target use case in this repository:
- prepared_nodes/<node>.csv with columns: timestamp, cpu, voltage
- online / replay anomaly scoring for a single node
- common metric schema for Prometheus / Grafana

Implemented here:
1) Exact z-normalized MASS distance profile (FFT-based), ported from the Matlab idea.
2) 2D discord score as sum of per-dimension distance profiles.
3) Backward iterative-doubling search inspired by Classic / Multidimensional DAMP.
4) Streaming detector that works on already arrived history only.
5) Optional offline helper that computes a left profile over a whole series.

Important note:
For the replay microservice we intentionally use the streaming-safe variant:
- score of the current subsequence is computed only from the past;
- no future leakage is used in the online path;
- forward pruning from the batch paper is left as a future optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-12


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two greater than or equal to x."""
    x = int(max(1, x))
    return 1 << (x - 1).bit_length()


def z_normalize(x: Sequence[float], eps: float = EPS) -> np.ndarray:
    """Z-normalize a 1D array with graceful handling of near-constant windows."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    if not np.isfinite(std) or std < eps:
        return np.zeros_like(arr, dtype=float)
    return (arr - mean) / std


def has_constant_region(series: Sequence[float], window: int, min_std: float = 1e-8) -> bool:
    """Check whether the whole series or any window is near-constant."""
    x = np.asarray(series, dtype=float).reshape(-1)
    if len(x) < window:
        return False
    if float(np.std(x)) < min_std:
        return True
    kernel = np.ones(window, dtype=float)
    sum_x = np.convolve(x, kernel, mode="valid")
    sum_x2 = np.convolve(x * x, kernel, mode="valid")
    mean = sum_x / window
    var = np.maximum(sum_x2 / window - mean * mean, 0.0)
    return bool(np.any(var < min_std * min_std))


def _sliding_mean_std(x: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return mean and std for all length-m windows of x."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if m <= 0 or len(x) < m:
        return np.array([], dtype=float), np.array([], dtype=float)

    csum = np.cumsum(np.insert(x, 0, 0.0))
    csum2 = np.cumsum(np.insert(x * x, 0, 0.0))
    sum_x = csum[m:] - csum[:-m]
    sum_x2 = csum2[m:] - csum2[:-m]
    mean = sum_x / m
    var = np.maximum(sum_x2 / m - mean * mean, 0.0)
    std = np.sqrt(var)
    std[std < EPS] = np.inf
    return mean, std


def mass_v2(x: Sequence[float], y: Sequence[float]) -> np.ndarray:
    """
    Exact z-normalized Euclidean distance profile, aligned with Matlab MASS_V2.

    Parameters
    ----------
    x : data segment where candidate windows are searched
    y : query subsequence

    Returns
    -------
    np.ndarray of length len(x) - len(y) + 1
    """
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    n = len(x_arr)
    m = len(y_arr)

    if m <= 0:
        raise ValueError("Query length must be positive")
    if n < m:
        return np.array([], dtype=float)

    meany = float(y_arr.mean())
    sigmay = float(y_arr.std(ddof=0))
    if not np.isfinite(sigmay) or sigmay < EPS:
        return np.full(n - m + 1, np.inf, dtype=float)

    meanx, sigmax = _sliding_mean_std(x_arr, m)

    k = 1 << (n + m - 1).bit_length()
    y_rev = y_arr[::-1]
    X = np.fft.fft(x_arr, k)
    Y = np.fft.fft(y_rev, k)
    z = np.fft.ifft(X * Y).real
    dot = z[m - 1 : n]

    denom = sigmax * sigmay
    valid = np.isfinite(denom) & (denom > EPS)
    corr_term = np.empty_like(dot)
    corr_term.fill(np.nan)
    corr_term[valid] = (dot[valid] - m * meanx[valid] * meany) / denom[valid]

    dist2 = 2.0 * (m - corr_term)
    dist2 = np.maximum(dist2, 0.0)
    dist = np.sqrt(dist2)
    dist[~valid] = np.inf
    return dist


def multidim_mass_v2(
    cpu_history: Sequence[float],
    cpu_query: Sequence[float],
    voltage_history: Sequence[float],
    voltage_query: Sequence[float],
) -> np.ndarray:
    """2D distance profile as sum of CPU and voltage MASS profiles."""
    d_cpu = mass_v2(cpu_history, cpu_query)
    d_voltage = mass_v2(voltage_history, voltage_query)
    if len(d_cpu) != len(d_voltage):
        raise ValueError("Dimension profiles must have the same length")
    return d_cpu + d_voltage


@dataclass
class DAMPResult:
    score: float
    is_exact: bool
    search_span: int
    position: int


class MultidimDAMPStreamDetector:
    """
    Streaming-safe 2D DAMP detector.

    The detector processes one aligned sample at a time. Once enough points have
    arrived to form a complete query window and the start index is >= start_index,
    it computes a left-discord score using only the past.

    Parameters are chosen to stay close to the Matlab implementation:
    - initial backward span = next_power_of_two(8 * window_size)
    - iterative doubling until a match below best-so-far is found or history ends
    """

    def __init__(
        self,
        window_size: int,
        start_index: Optional[int] = None,
        init_backward_factor: int = 8,
        tiny_decrement: float = 1e-5,
    ) -> None:
        if window_size <= 2:
            raise ValueError("window_size must be > 2")
        self.window_size = int(window_size)
        self.start_index = int(start_index) if start_index is not None else max(window_size, 4 * window_size)
        self.init_backward_factor = int(max(1, init_backward_factor))
        self.tiny_decrement = float(tiny_decrement)

        self.cpu: List[float] = []
        self.voltage: List[float] = []
        self.left_profile: List[float] = []
        self.best_so_far: float = float("-inf")

    @property
    def samples_seen(self) -> int:
        return len(self.cpu)

    @property
    def subsequences_seen(self) -> int:
        return max(0, self.samples_seen - self.window_size + 1)

    def _current_query_start(self) -> Optional[int]:
        if self.samples_seen < self.window_size:
            return None
        return self.samples_seen - self.window_size

    def update(self, cpu_value: float, voltage_value: float) -> Optional[DAMPResult]:
        self.cpu.append(float(cpu_value))
        self.voltage.append(float(voltage_value))

        query_start = self._current_query_start()
        if query_start is None:
            return None

        if query_start < self.start_index:
            score = 0.0 if not self.left_profile else max(0.0, self.left_profile[-1] - self.tiny_decrement)
            self.left_profile.append(score)
            if score > self.best_so_far:
                self.best_so_far = score
            return None

        result = self._score_at(query_start)
        self.left_profile.append(result.score)
        if result.score > self.best_so_far:
            self.best_so_far = result.score
        return result

    def _score_at(self, query_start: int) -> DAMPResult:
        m = self.window_size
        query_cpu = np.asarray(self.cpu[query_start : query_start + m], dtype=float)
        query_voltage = np.asarray(self.voltage[query_start : query_start + m], dtype=float)

        query_cpu = z_normalize(query_cpu)
        query_voltage = z_normalize(query_voltage)

        if query_start < m:
            return DAMPResult(score=float("inf"), is_exact=True, search_span=query_start, position=query_start)

        approx = float("inf")
        search_span = next_power_of_two(self.init_backward_factor * m)
        exact = False

        while approx >= self.best_so_far:
            if query_start - search_span < 0:
                hist_cpu = np.asarray(self.cpu[:query_start + m - 1], dtype=float)
                hist_voltage = np.asarray(self.voltage[:query_start + m - 1], dtype=float)
                # only candidate starts strictly before query_start and non-self by construction
                profile = multidim_mass_v2(hist_cpu, query_cpu, hist_voltage, query_voltage)
                exact_profile = profile[:query_start - m + 1] if query_start - m + 1 > 0 else np.array([], dtype=float)
                approx = float(np.min(exact_profile)) if exact_profile.size else float("inf")
                exact = True
                break

            seg_start = query_start - search_span
            seg_end = query_start + m - 1
            hist_cpu = np.asarray(self.cpu[seg_start:seg_end], dtype=float)
            hist_voltage = np.asarray(self.voltage[seg_start:seg_end], dtype=float)
            profile = multidim_mass_v2(hist_cpu, query_cpu, hist_voltage, query_voltage)
            approx = float(np.min(profile)) if profile.size else float("inf")
            if approx < self.best_so_far:
                break
            search_span *= 2

        return DAMPResult(score=approx, is_exact=exact, search_span=min(search_span, query_start), position=query_start)


def damp_multidim_left_profile(
    cpu: Sequence[float],
    voltage: Sequence[float],
    window_size: int,
    start_index: Optional[int] = None,
    init_backward_factor: int = 8,
) -> np.ndarray:
    """Offline helper: compute streaming-style left profile over the full series."""
    detector = MultidimDAMPStreamDetector(
        window_size=window_size,
        start_index=start_index,
        init_backward_factor=init_backward_factor,
    )
    for c, v in zip(cpu, voltage):
        detector.update(c, v)
    return np.asarray(detector.left_profile, dtype=float)


def recommend_start_index(window_size: int) -> int:
    """Recommended warm-up start from the paper / Matlab hints."""
    return max(window_size, 4 * window_size)


__all__ = [
    "DAMPResult",
    "MultidimDAMPStreamDetector",
    "damp_multidim_left_profile",
    "has_constant_region",
    "mass_v2",
    "multidim_mass_v2",
    "next_power_of_two",
    "recommend_start_index",
    "z_normalize",
]
