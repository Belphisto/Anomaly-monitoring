# coding: utf-8

import numpy as np


def _fallback_mp(ts, window):
    ts = np.asarray(ts, dtype=float)
    n = len(ts)
    k = n - window + 1
    if k <= 0:
        return {"mp": np.array([]), "pi": np.array([], dtype=int)}

    subs = np.array([ts[i:i + window] for i in range(k)], dtype=float)
    means = subs.mean(axis=1, keepdims=True)
    stds = subs.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    subs = (subs - means) / stds

    mp = np.full(k, np.inf, dtype=float)
    pi = np.full(k, -1, dtype=int)
    exclusion = max(1, window // 2)

    for i in range(k):
        dists = np.linalg.norm(subs - subs[i], axis=1)
        left = max(0, i - exclusion)
        right = min(k, i + exclusion + 1)
        dists[left:right] = np.inf
        j = int(np.argmin(dists))
        if np.isfinite(dists[j]):
            mp[i] = float(dists[j])
            pi[i] = j

    finite = mp[np.isfinite(mp)]
    if finite.size:
        fill = float(np.max(finite))
        mp[~np.isfinite(mp)] = fill
    else:
        mp[:] = 0.0

    return {"mp": mp, "pi": pi}


def find_mp(ts, m) -> dict:
    """Return Matrix Profile with compatibility across matrixprofile package variants."""

    window = max(4, m // 2)

    try:
        import matrixprofile as mp

        if hasattr(mp, "compute"):
            result = mp.compute(ts, windows=window)
            if isinstance(result, dict) and "mp" in result:
                return result

        if hasattr(mp, "analyze"):
            result = mp.analyze(ts, windows=window)
            if isinstance(result, dict) and "mp" in result:
                return result

        if hasattr(mp, "algorithms") and hasattr(mp.algorithms, "stomp"):
            result = mp.algorithms.stomp(ts, window)
            if isinstance(result, dict) and "mp" in result:
                return result
    except Exception:
        pass

    return _fallback_mp(ts, window)
