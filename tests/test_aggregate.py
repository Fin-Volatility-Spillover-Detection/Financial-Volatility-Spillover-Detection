"""Birleştirme ve kayan faktör modeli için garantiler."""

import numpy as np
import pandas as pd

from sektorgraf.data.aggregate import compound, ff3_aggregate, period_keys, realized_variance, with_min_max
from sektorgraf.data.residuals import rolling_factor_model, rolling_residuals


def _days(start="1984-12-24", end="1985-03-01"):  # Cuma biter
    return pd.bdate_range(start, end)


def test_compound_weekly_and_label_is_last_trading_day():
    idx = _days()
    df = pd.DataFrame({"A": 1.0}, index=idx)          # her gün %1
    w = compound(df, "haftalik")
    assert all(w.index.day_name() == "Friday")
    assert np.isclose(w["A"].iloc[1], (1.01 ** 5 - 1) * 100)   # 31 Ara 1984 – 4 Oca 1985 tam hafta
    # Cuma tatilse etiket Perşembe olur
    idx2 = idx.drop(pd.Timestamp("1985-01-18"))
    w2 = compound(pd.DataFrame({"A": 1.0}, index=idx2), "haftalik")
    assert pd.Timestamp("1985-01-17") in w2.index


def test_biweekly_anchor():
    keys = period_keys(_days(), "2haftalik")
    ends = sorted(set(pd.DatetimeIndex(keys)))
    assert pd.Timestamp("1985-01-11") in ends and pd.Timestamp("1985-01-25") in ends
    assert pd.Timestamp("1985-01-18") not in ends


def test_monthly_compound():
    idx = _days("1985-01-01", "1985-03-31")
    df = pd.DataFrame({"A": np.linspace(-1, 1, len(idx))}, index=idx)
    m = compound(df, "aylik")
    jan = df.loc["1985-01", "A"]
    assert np.isclose(m["A"].iloc[0], ((1 + jan / 100).prod() - 1) * 100)


def test_min_max_columns_order():
    df = pd.DataFrame({"A": [1.0, -2.0, 3.0, 0.5, 0.0]}, index=_days("1985-01-07", "1985-01-11"))
    out = with_min_max(df, "haftalik")
    assert list(out.columns) == ["A", "A_min", "A_max"]
    assert out["A_min"].iloc[0] == -2.0 and out["A_max"].iloc[0] == 3.0


def test_ff3_market_excess_is_not_compounded_directly():
    idx = _days("1985-01-07", "1985-01-11")
    ff = pd.DataFrame({"Mkt-RF": 1.0, "SMB": 0.0, "HML": 0.0, "RF": 0.5}, index=idx)
    out = ff3_aggregate(ff, "haftalik")
    mkt = (1.015 ** 5 - 1) * 100
    rf = (1.005 ** 5 - 1) * 100
    assert np.isclose(out["Mkt-RF"].iloc[0], mkt - rf)
    assert out["n_gun"].iloc[0] == 5


def test_realized_variance():
    idx = _days("1985-01-07", "1985-01-11")
    d = pd.DataFrame({"A": [1.0, -2.0, 0.0, 1.0, 2.0]}, index=idx)
    rv = realized_variance(d, "haftalik")
    assert rv["A"].iloc[0] == 10.0 and rv["n_gun"].iloc[0] == 5


def test_factor_model_no_lookahead_all_coefficients():
    rng = np.random.default_rng(0)
    T = 600
    idx = pd.bdate_range("1990-01-01", periods=T)
    ff = pd.DataFrame(rng.normal(size=(T, 3)), idx, ["Mkt-RF", "SMB", "HML"]).assign(RF=0.01)
    sec = pd.DataFrame(ff[["Mkt-RF", "SMB", "HML"]].to_numpy() @ rng.normal(size=(3, 4))
                       + rng.normal(size=(T, 4)), idx, list("abcd"))
    r1, c1 = rolling_factor_model(sec, ff, "ff3", 104, 84)
    sec2 = sec.copy()
    sec2.iloc[400:] *= 50
    r2, c2 = rolling_factor_model(sec2, ff, "ff3", 104, 84)
    pd.testing.assert_frame_equal(r1.iloc[:400], r2.iloc[:400])
    for k in c1:
        pd.testing.assert_frame_equal(c1[k].iloc[:401], c2[k].iloc[:401])  # t=400 katsayısı t<400'den
    # geriye uyumlu kısa yol aynı sonucu verir
    r3, b3 = rolling_residuals(sec, ff, "ff3", 104, 84)
    pd.testing.assert_frame_equal(r1, r3)
    pd.testing.assert_frame_equal(c1["Mkt-RF"], b3)


def test_factor_model_recovers_known_betas():
    """Bilinen betalarla üretilmiş veride kayan pencere betaları doğru bulunmalı."""
    rng = np.random.default_rng(1)
    T = 800
    idx = pd.bdate_range("1990-01-01", periods=T)
    ff = pd.DataFrame(rng.normal(size=(T, 3)), idx, ["Mkt-RF", "SMB", "HML"]).assign(RF=0.0)
    true = np.array([[1.2, 0.3, 0.8], [0.6, -0.2, 0.4]])  # 2 sektör × (mkt, smb, hml)
    sec = pd.DataFrame(ff[["Mkt-RF", "SMB", "HML"]].to_numpy() @ true.T
                       + rng.normal(0, 0.5, (T, 2)), idx, ["A", "B"])
    _, coefs = rolling_factor_model(sec, ff, "ff3", 252, 200)
    last = np.array([[coefs[k].iloc[-1][s] for k in ("Mkt-RF", "SMB", "HML")] for s in ("A", "B")])
    assert np.allclose(last, true, atol=0.1)
