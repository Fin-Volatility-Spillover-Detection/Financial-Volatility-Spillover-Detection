"""Metrikler. Girdiler: aynı şekilli (gün × sektör) tahmin ve gerçek tabloları."""

import numpy as np
import pandas as pd
from scipy import stats


def rmse(pred: pd.DataFrame, actual: pd.DataFrame) -> float:
    return float(np.sqrt(np.nanmean((pred - actual).to_numpy() ** 2)))


def mae(pred, actual) -> float:
    return float(np.nanmean(np.abs((pred - actual).to_numpy())))


def r2_oos(pred, actual, benchmark=None) -> float:
    """Örneklem dışı R² = 1 − SSE_model / SSE_benchmark. Benchmark verilmezse sıfır tahmini.

    Pozitif = benchmark'tan iyi. Günlük getiride %0,5 bile kayda değer.
    """
    bench = 0.0 if benchmark is None else benchmark.to_numpy()
    a = actual.to_numpy()
    return float(1 - np.nansum((a - pred.to_numpy()) ** 2) / np.nansum((a - bench) ** 2))


def direction_accuracy(pred, actual) -> float:
    """Yönü (artı/eksi) doğru bilme oranı. 0,50 = yazı tura. Hep 0 tahmin eden model için NaN."""
    p, a = np.sign(pred.to_numpy()), np.sign(actual.to_numpy())
    m = (a != 0) & (p != 0) & ~np.isnan(a) & ~np.isnan(p)
    return float((p[m] == a[m]).mean()) if m.any() else float("nan")


def diebold_mariano(pred_a, pred_b, actual, horizon: int = 1, power: int = 2) -> dict:
    """A ile B'nin tahmin hataları arasındaki fark şans eseri mi?

    Panel: her gün için kayıp 49 sektörün ortalaması alınır, sonra günlük seri üzerinde test.
    d_t = kayıp_A − kayıp_B. Negatif ortalama → A daha iyi.
    Harvey–Leybourne–Newbold küçük örneklem düzeltmesi uygulanır.
    """
    la = (np.abs(pred_a - actual) ** power).mean(axis=1)
    lb = (np.abs(pred_b - actual) ** power).mean(axis=1)
    d = (la - lb).dropna().to_numpy()
    T = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[: T - k] - dbar)) / T for k in range(horizon)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / T
    dm = dbar / np.sqrt(var)
    hln = np.sqrt((T + 1 - 2 * horizon + horizon * (horizon - 1) / T) / T)
    stat = dm * hln
    return {
        "stat": float(stat),
        "p_two_sided": float(2 * stats.t.sf(abs(stat), df=T - 1)),
        "p_a_better": float(stats.t.cdf(stat, df=T - 1)),  # H1: A'nın kaybı daha küçük
        "mean_loss_diff": float(dbar),
        "n_days": T,
    }
