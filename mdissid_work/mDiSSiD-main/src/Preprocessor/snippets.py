# coding: utf-8

"""
Совместимая версия snippets.py для mDiSSiD.

Причина замены:
исходный файл использует приватную функцию stumpy.mpdist._mpdist_vect,
сигнатура которой изменилась в новых версиях stumpy. Из-за этого preprocessor.py
падает на этапе поиска snippets.

Эта версия использует только публичный API stumpy и сохраняет тот же внешний
интерфейс функций, который ожидает репозиторий mDiSSiD.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import stumpy
from stumpy import core
from stumpy.aampdist_snippets import aampdist_snippets


def _normalize_snippets_result(result: Any):
    """Привести результат stumpy.snippets к ожидаемому кортежу из 6 элементов."""
    if isinstance(result, tuple) and len(result) >= 6:
        return result[:6]

    if isinstance(result, dict):
        return (
            np.asarray(result.get("snippets", []), dtype=float),
            np.asarray(result.get("indices", []), dtype=np.int64),
            np.asarray(result.get("profiles", []), dtype=float),
            np.asarray(result.get("fractions", []), dtype=float),
            np.asarray(result.get("areas", []), dtype=float),
            np.asarray(result.get("regimes", []), dtype=np.int64),
        )

    raise TypeError(f"Неподдерживаемый формат результата stumpy.snippets: {type(result)!r}")


def _safe_stumpy_snippets(T, m, k, s=None, mpdist_percentage=0.05):
    """Вызов stumpy.snippets с совместимостью по версиям библиотеки."""
    call_variants = []

    kwargs = {"m": m, "k": k}
    if s is not None:
        kwargs["s"] = s
    if mpdist_percentage is not None:
        kwargs["mpdist_percentage"] = mpdist_percentage
    call_variants.append(kwargs)

    kwargs2 = {"m": m, "k": k}
    if s is not None:
        kwargs2["s"] = s
    call_variants.append(kwargs2)

    for kw in call_variants:
        try:
            return _normalize_snippets_result(stumpy.snippets(T, **kw))
        except TypeError:
            continue

    # Последняя попытка — минимальный публичный вызов
    return _normalize_snippets_result(stumpy.snippets(T, m=m, k=k))


@core.non_normalized(aampdist_snippets)
def find_snippets(
    T,
    m,
    k,
    percentage=1.0,
    s=None,
    mpdist_percentage=0.05,
    mpdist_k=None,
    normalize=True,
    p=2.0,
):
    """
    Совместимая замена исходной реализации.

    Вместо ручного вычисления MPdist-профилей через приватный _mpdist_vect
    используется публичная функция stumpy.snippets.
    """
    T = np.asarray(T, dtype=float)
    T = stumpy.core._preprocess(T)

    if m > T.shape[0] // 2:
        raise ValueError(
            f"The snippet window size of {m} is too large for a time series with "
            f"length {T.shape[0]}. Please try `m <= len(T) // 2`."
        )

    if s is not None:
        s = min(int(s), m)
    else:
        percentage = np.clip(percentage, 0.0, 1.0)
        s = min(int(math.ceil(percentage * m)), m)

    return _safe_stumpy_snippets(T=T, m=m, k=k, s=s, mpdist_percentage=mpdist_percentage)


def find_mpdist_percentage(m, l) -> float:
    default_k_percentage = 0.05
    k_percentage = 1 + (1 - 2 * l) / (2 * m - 2 * l + 2)
    k_percentage = max(default_k_percentage, k_percentage)
    return float(min(max(k_percentage, 0.05), 1.0))


def find_snippets_with_optimization(ts, m, l, snippets_num) -> dict:
    """Return snippets in time series.

    В текущей совместимой версии используется публичный stumpy.snippets.
    Оптимизационный шаг исходной статьи здесь не воспроизводится, но внешний
    формат результата полностью сохранён.
    """
    k_percentage = find_mpdist_percentage(m, l)
    result = find_snippets(T=ts, m=m, k=snippets_num, s=l, mpdist_percentage=k_percentage)

    return {
        "snippets": result[0],
        "indices": result[1],
        "profiles": result[2],
        "fractions": result[3],
        "areas": result[4],
        "regimes": result[5],
    }


def find_snippets_without_optimization(ts, m, l, snippets_num) -> dict:
    """Return snippets using the public STUMPY implementation directly."""
    k_percentage = find_mpdist_percentage(m, l)
    result = _safe_stumpy_snippets(T=np.asarray(ts, dtype=float), m=m, k=snippets_num, s=l, mpdist_percentage=k_percentage)

    return {
        "snippets": result[0],
        "indices": result[1],
        "profiles": result[2],
        "fractions": result[3],
        "areas": result[4],
        "regimes": result[5],
    }


def find_profiles_curve(profiles, snippets_num) -> list:
    profiles_curve = []
    for i in range(len(profiles[0])):
        min_value = float("inf")
        for j in range(snippets_num):
            if min_value > profiles[j][i]:
                min_value = profiles[j][i]
        profiles_curve.append(min_value)
    return profiles_curve


def find_mpdist_regimes(regimes, profiles, snippets_num):
    mpdist_regimes = []

    for snippet_label in range(snippets_num):
        mpdist_regime = []
        mpdist_regime_idx = []

        for j in range(len(regimes)):
            regime_label = regimes[j][0]
            if snippet_label == regime_label:
                start_regime = regimes[j][1]
                end_regime = regimes[j][2]
                mpdist_regime.extend(profiles[snippet_label][start_regime:end_regime])
                mpdist_regime_idx.extend(np.arange(start_regime, end_regime))

        mpdist_regimes.append(pd.Series(mpdist_regime, index=mpdist_regime_idx))

    return mpdist_regimes


def find_regimes_profiles(mpdist_regimes, n, snippets_num):
    regimes_profiles = []

    for i in range(snippets_num):
        regimes_profiles.append([0] * n)
        keys = list(mpdist_regimes[i].keys())
        vals = list(mpdist_regimes[i].values)
        for j in range(len(keys)):
            regimes_profiles[i][keys[j]] = vals[j]

    return regimes_profiles


def find_mpdist_all_regimes(regimes, profiles, snippets_num):
    mpdist_regimes = []

    for snippet_label in range(snippets_num):
        mpdist_regime = []

        for j in range(len(regimes)):
            regime_label = regimes[j][0]
            if snippet_label == regime_label:
                start_regime = regimes[j][1]
                end_regime = regimes[j][2]
                mpdist_regime.extend(profiles[snippet_label][start_regime:end_regime])

        mpdist_regimes.append(mpdist_regime)

    return mpdist_regimes
