"""Kritik garantiler: look-ahead yok, metrikler doğru, graf yöntemleri bilinen grafı buluyor."""

import numpy as np
import pandas as pd
import pytest

from sektorgraf.data.residuals import rolling_residuals
from sektorgraf.evaluation import metrics as M
from sektorgraf.graphs import dy, granger
from sektorgraf.graphs.utils import random_like
from sektorgraf.models.baselines import AR


def _synthetic(T=3000, seed=0):
    """Bilinen graf: A → B (gecikme 1, katsayı 0.4). C bağımsız. Ortak piyasa faktörü var."""
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0, 1, T)
    e = rng.normal(0, 1, (T, 3))
    r = np.zeros((T, 3))
    r[:, 0] = e[:, 0]
    r[1:, 1] = 0.4 * r[:-1, 0] + e[1:, 1]
    r[:, 2] = e[:, 2]
    idx = pd.bdate_range("2000-01-03", periods=T)
    resid = pd.DataFrame(r, idx, ["A", "B", "C"])
    betas = np.array([0.8, 1.2, 1.0])
    sectors = resid + np.outer(mkt, betas)
    ff3 = pd.DataFrame({"Mkt-RF": mkt, "SMB": 0.0, "HML": 0.0, "RF": 0.0}, idx)
    return sectors, ff3, resid


def test_residuals_no_lookahead():
    """Geleceği değiştirmek geçmişin artıklarını değiştirmemeli."""
    sectors, ff3, _ = _synthetic()
    r1, _ = rolling_residuals(sectors, ff3, "capm", window=252, min_obs=200)
    s2 = sectors.copy()
    s2.iloc[2000:] *= 10
    r2, _ = rolling_residuals(s2, ff3, "capm", window=252, min_obs=200)
    pd.testing.assert_frame_equal(r1.iloc[:2000], r2.iloc[:2000])


def test_residuals_recover_beta():
    sectors, ff3, _ = _synthetic()
    _, betas = rolling_residuals(sectors, ff3, "capm", window=252, min_obs=200)
    assert np.allclose(betas.iloc[-1], [0.8, 1.2, 1.0], atol=0.15)


def test_granger_finds_known_edge():
    _, _, resid = _synthetic()
    _, _, sig = granger.granger_graph(resid, lags=1, fdr=0.05)
    assert sig.loc["B", "A"]           # A → B bulunmalı
    assert sig.values.sum() == 1       # başka kenar olmamalı


def test_dy_rows_sum_to_100_and_direction():
    _, _, resid = _synthetic()
    theta = dy.dy_table(resid, lags=1, horizon=10, estimator="ols")
    assert np.allclose(theta.sum(axis=1), 100)
    assert theta.loc["B", "A"] > 5 * theta.loc["A", "B"]   # etki A'dan B'ye
    assert dy.connectedness(theta)["net"]["A"] > 0          # A net şok verici


def test_random_graph_keeps_edge_count():
    W = pd.DataFrame(np.array([[0, 2, 0], [1, 0, 0], [0, 3, 0]], float), list("abc"), list("abc"))
    R = random_like(W, seed=1)
    assert (R.values != 0).sum() == 3
    assert sorted(R.values[R.values != 0]) == [1, 2, 3]
    assert np.all(np.diag(R.values) == 0)


def test_ar_uses_only_past():
    _, _, resid = _synthetic()
    m = AR(1).fit(resid.iloc[:2000])
    d = resid.index[2500]
    p1 = m.predict(resid, [d])
    changed = resid.copy()
    changed.loc[d:] += 100   # tahmin günü ve sonrası değişirse tahmin değişmemeli
    pd.testing.assert_frame_equal(p1, m.predict(changed, [d]))


def test_metrics():
    a = pd.DataFrame({"x": [1.0, -1.0, 2.0, -2.0]})
    assert M.rmse(a, a) == 0
    assert M.r2_oos(a, a) == 1
    assert M.r2_oos(a * 0, a) == 0
    assert M.direction_accuracy(a, a) == 1


def test_dm_detects_better_model():
    rng = np.random.default_rng(0)
    actual = pd.DataFrame(rng.normal(size=(1000, 5)))
    good = actual + rng.normal(0, 0.5, actual.shape)
    bad = actual + rng.normal(0, 1.0, actual.shape)
    assert M.diebold_mariano(good, bad, actual)["p_a_better"] < 0.01
