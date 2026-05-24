#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

try:
    from kneed import KneeLocator
except Exception:
    KneeLocator = None


def _to_1d_array(x):
    if isinstance(x, pd.Series):
        arr = x.to_numpy(dtype=float)
    elif isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            arr = np.array([], dtype=float)
        else:
            arr = x.iloc[:, 0].to_numpy(dtype=float)
    else:
        arr = np.asarray(x, dtype=float)

    arr = np.ravel(arr).astype(float, copy=False)
    if arr.size == 0:
        return arr

    finite = np.isfinite(arr)
    if not finite.all():
        if finite.any():
            fill = float(np.nanmedian(arr[finite]))
            arr = np.where(finite, arr, fill)
        else:
            arr = np.zeros_like(arr, dtype=float)
    return arr


def _extract_index(x, n):
    if isinstance(x, pd.Series):
        try:
            idx = np.asarray(x.index, dtype=int)
            if idx.shape[0] == n:
                return idx
        except Exception:
            pass

    if isinstance(x, pd.DataFrame):
        try:
            idx = np.asarray(x.index, dtype=int)
            if idx.shape[0] == n:
                return idx
        except Exception:
            pass

    return np.arange(n, dtype=int)


def _safe_cluster_count(n_samples, requested_max=3):
    if n_samples <= 1:
        return 1
    return max(1, min(int(requested_max), int(n_samples)))


def _choose_k(values_2d):
    n = values_2d.shape[0]
    max_k = _safe_cluster_count(n, 3)
    if max_k <= 1:
        return 1

    inertias = []
    ks = list(range(1, max_k + 1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(values_2d)
        inertias.append(float(km.inertia_))

    if KneeLocator is not None and len(ks) >= 3:
        try:
            kn = KneeLocator(ks, inertias, curve="convex", direction="decreasing")
            if kn.elbow is not None:
                return int(kn.elbow)
        except Exception:
            pass

    return 2 if max_k >= 2 else 1


def _find_snippet_local_idx(regime_index, snippet_global_idx):
    if regime_index.size == 0:
        return 0

    exact = np.where(regime_index == int(snippet_global_idx))[0]
    if exact.size > 0:
        return int(exact[0])

    return int(np.argmin(np.abs(regime_index.astype(int) - int(snippet_global_idx))))


def find_snippets_anomalies_KNN(regimes_profiles, snippets_indices, N, snippets_num):
    print("KNN")

    results = {"indices": []}
    if regimes_profiles is None:
        return results

    snippets_indices = list(snippets_indices)
    total = min(int(snippets_num), len(regimes_profiles), len(snippets_indices))

    for i in range(total):
        raw_profile = regimes_profiles[i]
        profile = _to_1d_array(raw_profile)
        if profile.size == 0:
            continue

        regime_index = _extract_index(raw_profile, profile.size)
        values_2d = profile.reshape(-1, 1)

        k = _choose_k(values_2d)
        print(k)

        if k <= 1:
            continue

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(values_2d)
        centers = km.cluster_centers_.reshape(-1)

        snippet_local_idx = _find_snippet_local_idx(regime_index, int(snippets_indices[i]))
        snippet_cluster = int(labels[snippet_local_idx])

        dists = np.abs(centers - centers[snippet_cluster])
        dists[snippet_cluster] = -1.0
        anomaly_cluster = int(np.argmax(dists))

        anomaly_positions = np.where(labels == anomaly_cluster)[0]
        if anomaly_positions.size == 0:
            continue

        start = anomaly_positions[0]
        prev = anomaly_positions[0]

        for pos in anomaly_positions[1:]:
            if int(pos) == int(prev) + 1:
                prev = pos
                continue

            results["indices"].append([
                int(i),
                int(regime_index[start]),
                int(regime_index[prev]),
            ])
            start = pos
            prev = pos

        results["indices"].append([
            int(i),
            int(regime_index[start]),
            int(regime_index[prev]),
        ])

    return results


def construct_snippets_anomalies_annotation(
    max_mpdist_regimes,
    ts_snippets_anomalies,
    snippets_indices,
    n,
    snippets_num,
    m,
):
    annotation = [1] * int(n)

    if not ts_snippets_anomalies:
        return annotation

    indices = ts_snippets_anomalies.get("indices", [])
    if indices is None:
        return annotation

    for item in indices:
        if item is None or len(item) < 3:
            continue

        _, start_idx, end_idx = item

        try:
            start_idx = int(start_idx)
            end_idx = int(end_idx)
        except Exception:
            continue

        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx

        start_idx = max(0, start_idx)
        end_idx = min(int(n) - 1, end_idx)

        for idx in range(start_idx, end_idx + 1):
            annotation[idx] = -1

    return annotation

def construct_anomalies_annotation(discords_annotation, snippets_anomalies_annotation):
    """
    Объединяет две аннотации:
    1  -> норма
    -1 -> аномалия

    Если хотя бы в одной аннотации точка помечена как аномальная,
    в итоговой аннотации она тоже считается аномальной.
    """
    n = min(len(discords_annotation), len(snippets_anomalies_annotation))
    result = [1] * n

    for i in range(n):
        d = discords_annotation[i]
        s = snippets_anomalies_annotation[i]

        if int(d) == -1 or int(s) == -1:
            result[i] = -1
        else:
            result[i] = 1

    return result