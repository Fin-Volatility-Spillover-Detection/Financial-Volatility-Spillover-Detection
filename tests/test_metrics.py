"""Ortak metriklerin bilinen değerlerle doğruluğu."""

import numpy as np
import pandas as pd
import pytest

from sektorgraf.evaluation.metrics import (
    check_aligned,
    diebold_mariano,
    direction_accuracy,
    oos_r2,
    rmse,
    rmse_by_sector,
)


def _frame(values, cols=("A", "B")):
    idx = pd.date_range("2019-01-04", periods=len(values), freq="W-FRI", name="Date")
    return pd.DataFrame(values, index=idx, columns=list(cols), dtype=float)


ACTUAL = _frame([[1, -2], [3, 0], [-1, 2], [2, -1]])
ZERO = ACTUAL * 0


def test_perfect_prediction():
    assert rmse(ACTUAL, ACTUAL) == 0
    assert oos_r2(ACTUAL, ACTUAL) == 1
    assert direction_accuracy(ACTUAL, ACTUAL) == 1


def test_zero_model():
    assert np.isclose(rmse(ZERO, ACTUAL), np.sqrt(np.mean(ACTUAL.values ** 2)))
    assert oos_r2(ZERO, ACTUAL) == 0
    assert np.isnan(direction_accuracy(ZERO, ACTUAL))


def test_known_values():
    pred = ACTUAL + 1                                    # her hata 1
    assert np.isclose(rmse(pred, ACTUAL), 1)
    sse_zero = np.sum(ACTUAL.values ** 2)                # 24
    assert np.isclose(oos_r2(pred, ACTUAL), 1 - 8 / sse_zero)
    by_sector = rmse_by_sector(_frame([[1, -2], [3, 2], [-1, 2], [2, -1]]), ACTUAL)
    assert by_sector["A"] == 0 and np.isclose(by_sector["B"], 1)   # B'de tek hata: sqrt(4/4)


def test_direction_excludes_actual_zero_and_counts_pred_zero_wrong():
    pred = _frame([[1, 1], [-5, 9], [-1, 0], [2, -1]])
    # gerçek 0 olan hücre hariç: 7 gözlem; yanlışlar: (1 / -2), (-5 / 3), (0 / 2) → doğru 4/7
    assert np.isclose(direction_accuracy(pred, ACTUAL), 4 / 7)


def test_alignment_errors():
    with pytest.raises(ValueError):
        check_aligned(ACTUAL[["B", "A"]], ACTUAL)        # sütun sırası farklı
    with pytest.raises(ValueError):
        check_aligned(ACTUAL.iloc[1:], ACTUAL)           # tarih farklı
    bad = ACTUAL.copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        check_aligned(bad, ACTUAL)


def test_diebold_mariano_known_value_and_sign():
    actual = _frame(np.zeros((4, 2)))
    # A'nın haftalık kaybı: (1+1)/2=1, 4, 1, 4 · B'nin: 0 → d = [1, 4, 1, 4]
    a = _frame([[1, 1], [2, 2], [1, -1], [2, -2]])
    res = diebold_mariano(a, actual, actual)
    d = np.array([1, 4, 1, 4.0])
    T = 4
    dm = d.mean() / np.sqrt(np.var(d) / T)               # gamma0, 1/T ile
    expected = dm * np.sqrt((T + 1 - 2) / T)             # HLN, h=1
    assert np.isclose(res["stat"], expected) and res["stat"] > 0     # B daha iyi → pozitif
    assert np.isclose(res["mean_diff"], 2.5) and res["n"] == 4
    swapped = diebold_mariano(actual, a, actual)
    assert np.isclose(swapped["stat"], -res["stat"]) and np.isclose(swapped["pvalue"], res["pvalue"])


def test_diebold_mariano_identical_models_undefined():
    res = diebold_mariano(ACTUAL + 1, ACTUAL + 1, ACTUAL)
    assert np.isnan(res["stat"]) and res["mean_diff"] == 0
