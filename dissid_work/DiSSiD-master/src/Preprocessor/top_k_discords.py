# coding: utf-8

import numpy as np


def find_discords(matrix_profile, m, discords_num) -> dict:
    """
    Return top-k discords based on the matrix profile.

    For a standard matrix profile, discords correspond to subsequences
    with the largest matrix profile values.
    """
    mp = np.asarray(matrix_profile["mp"], dtype=float)

    valid_idx = np.where(np.isfinite(mp))[0]
    if len(valid_idx) == 0:
        return {"discords": np.array([], dtype=int)}

    # Берем индексы с максимальными значениями mp
    order = valid_idx[np.argsort(mp[valid_idx])[::-1]]

    k = min(discords_num, len(order))
    discords = order[:k]

    return {"discords": discords}


def construct_discords_annotation(discords, n, m):
    discords_annotation = [1] * n
    anomaly_class = -1

    for i in range(len(discords)):
        discords_idxs = np.arange(discords[i] - int(m), discords[i] + int(m))
        discords_idxs = discords_idxs[discords_idxs >= 0]
        discords_idxs = discords_idxs[discords_idxs < n]

        for idx in discords_idxs:
            discords_annotation[idx] = anomaly_class

    return discords_annotation