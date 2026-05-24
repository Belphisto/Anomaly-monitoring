# coding: utf-8

import numpy as np

import config


def _extract_mp_array(matrix_profile):
    if isinstance(matrix_profile, dict):
        if "mp" in matrix_profile:
            return np.asarray(matrix_profile["mp"], dtype=float)
        if "matrix_profile" in matrix_profile:
            return np.asarray(matrix_profile["matrix_profile"], dtype=float)
    return np.asarray(matrix_profile, dtype=float)


def find_discords(matrix_profile, m, discords_num) -> dict:
    'Return Top-k discords with compatibility across matrixprofile variants.'

    profile = _extract_mp_array(matrix_profile)
    if profile.size == 0 or discords_num <= 0:
        return {"discords": np.array([], dtype=int)}

    try:
        import matrixprofile as mp
        if hasattr(mp, "discover") and hasattr(mp.discover, "discords"):
            discords = mp.discover.discords(matrix_profile, m // 2, k=discords_num)
            if isinstance(discords, dict) and "discords" in discords:
                return discords
    except Exception:
        pass

    work = profile.astype(float).copy()
    exclusion = max(1, int(m))
    picked = []

    for _ in range(min(discords_num, len(work))):
        idx = int(np.argmax(work))
        if not np.isfinite(work[idx]):
            break
        picked.append(idx)
        left = max(0, idx - exclusion)
        right = min(len(work), idx + exclusion + 1)
        work[left:right] = -np.inf

    return {"discords": np.asarray(picked, dtype=int)}


def construct_discords_annotation(discords, n, m):

    discords_annotation = [1]*n
    anomaly_class = -1

    for i in range(len(discords)):
        discords_idxs = np.arange(discords[i]-int(m), discords[i]+int(m))
        discords_idxs = discords_idxs[discords_idxs >= 0]
        discords_idxs = discords_idxs[discords_idxs < n]

        for idx in discords_idxs:
            discords_annotation[idx] = anomaly_class

    return discords_annotation
