"""Zero ve AR için garantiler: doğru katsayılar, look-ahead yok, format doğru."""

import numpy as np
import pandas as pd
import pytest

from sektorgraf.models.ar import fit_ar, predict_ar
from sektorgraf.models.zero import predict_zero


def _ar2_series(n=5000, seed=0):
    """Bilinen katsayılarla iki sektörlük yapay seri: A ~ AR(2), B ~ AR(1)."""
    rng = np.random.default_rng(seed)
    a, b = np.zeros(n), np.zeros(n)
    for t in range(2, n):
        a[t] = 0.1 + 0.3 * a[t - 1] - 0.2 * a[t - 2] + rng.normal()
        b[t] = -0.5 * b[t - 1] + rng.normal()
    idx = pd.date_range("1985-01-04", periods=n, freq="W-FRI", name="Date")
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def test_fit_recovers_known_coefficients():
    coefs = fit_ar(_ar2_series(), p=2)
    assert list(coefs.index) == ["c", "lag1", "lag2"] and list(coefs.columns) == ["A", "B"]
    np.testing.assert_allclose(coefs["A"], [0.1, 0.3, -0.2], atol=0.05)
    np.testing.assert_allclose(coefs["B"], [0.0, -0.5, 0.0], atol=0.05)


def test_prediction_matches_formula():
    df = _ar2_series(100)
    coefs = fit_ar(df.iloc[:60], p=2)
    pred = predict_ar(df, coefs, df.index[60:])
    t = df.index[70]
    expected = coefs.loc["c"] + coefs.loc["lag1"] * df.iloc[69] + coefs.loc["lag2"] * df.iloc[68]
    np.testing.assert_allclose(pred.loc[t], expected)


def test_no_look_ahead():
    """t'nin tahmini t ve sonrasındaki değerler değiştirildiğinde değişmemeli."""
    df = _ar2_series(100)
    train, dates = df.iloc[:60], df.index[60:]
    coefs = fit_ar(train, p=3)
    base = predict_ar(df, coefs, dates)
    t = 75
    changed = df.copy()
    changed.iloc[t:] = 999.0                       # t ve sonrası bozuldu
    after = predict_ar(changed, coefs, dates)
    pd.testing.assert_frame_equal(base.loc[: df.index[t]], after.loc[: df.index[t]])   # t dahil aynı
    assert not np.allclose(base.loc[df.index[t + 1]], after.loc[df.index[t + 1]])
    # eğitim katsayıları da sadece eğitim verisine bağlı
    pd.testing.assert_frame_equal(coefs, fit_ar(changed.iloc[:60], p=3))


def test_not_enough_history_raises():
    df = _ar2_series(100)
    coefs = fit_ar(df, p=2)
    with pytest.raises(ValueError):
        predict_ar(df, coefs, df.index[:5])


def test_zero_format():
    df = _ar2_series(10)
    z = predict_zero(df)
    assert z.index.equals(df.index) and list(z.columns) == list(df.columns)
    assert (z.values == 0).all()
