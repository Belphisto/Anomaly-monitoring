# coding: utf-8

import numpy as np
import stumpy


def find_mp(ts, m) -> dict:
    """
    Return Matrix Profile in the format expected by DiSSiD.

    DiSSiD expects:
        mp_result['mp'] -> 1D array of matrix profile values

    We compute the classic self-join matrix profile with STUMPY.
    """
    ts = np.asarray(ts, dtype=float)

    # В исходном коде DiSSiD использовалось окно m//2 для matrix profile
    # Оставляем это поведение, чтобы не ломать остальную логику пайплайна.
    window = max(4, m // 2)

    # STUMPY returns an array with columns:
    # [matrix_profile, profile_index, left_profile_index, right_profile_index]
    mp_raw = stumpy.stump(ts, m=window)

    mp_values = mp_raw[:, 0].astype(float)

    return {
        "mp": mp_values,
        "pi": mp_raw[:, 1],
        "lpi": mp_raw[:, 2],
        "rpi": mp_raw[:, 3],
        "window": window,
    }