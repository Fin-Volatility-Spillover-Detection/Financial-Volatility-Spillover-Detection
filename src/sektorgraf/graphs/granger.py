"""İkili Granger testleri + Benjamini–Hochberg FDR düzeltmesi → yönlü "öncülük grafı".

Her (j → i) çifti için:
  kısıtlı model    : r_i,t ~ r_i,t-1..t-p
  kısıtsız model   : r_i,t ~ r_i,t-1..t-p + r_j,t-1..t-p
  F testi          : j'nin gecikmeleri i'yi açıklamaya bir şey katıyor mu?
Terminoloji: raporda "nedensellik" değil, "öncülük" / "yönlü bağlantı".
"""

import numpy as np
import pandas as pd
from scipy import stats


def _lags(x: np.ndarray, p: int) -> np.ndarray:
    """(T,) → (T-p, p) gecikme matrisi; sütun k = x_{t-k-1}."""
    return np.column_stack([x[p - k - 1 : len(x) - k - 1] for k in range(p)])


def _rss(X: np.ndarray, y: np.ndarray) -> float:
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    return float(e @ e)


def benjamini_hochberg(pvals: np.ndarray, q: float) -> np.ndarray:
    """FDR q seviyesinde hangi testlerin anlamlı olduğunu (bool) döndürür."""
    p = np.asarray(pvals).ravel()
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    passed = ranked <= q
    sig = np.zeros(len(p), dtype=bool)
    if passed.any():
        sig[order[: np.max(np.where(passed)[0]) + 1]] = True
    return sig.reshape(np.shape(pvals))


def granger_graph(data: pd.DataFrame, lags: int = 1, fdr: float = 0.05):
    """Dönüş: (pvals, F, anlamlı_mı) — üçü de N×N DataFrame, [i, j] = j → i."""
    data = data.dropna()
    Y = data.to_numpy()
    T, N = Y.shape
    n = T - lags
    ones = np.ones((n, 1))
    lagged = [_lags(Y[:, k], lags) for k in range(N)]

    pvals = np.full((N, N), np.nan)
    fstat = np.full((N, N), np.nan)
    df_num, df_den = lags, n - 2 * lags - 1
    for i in range(N):
        y = Y[lags:, i]
        Xr = np.hstack([ones, lagged[i]])
        rss_r = _rss(Xr, y)
        for j in range(N):
            if i == j:
                continue
            rss_u = _rss(np.hstack([Xr, lagged[j]]), y)
            F = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
            fstat[i, j] = F
            pvals[i, j] = stats.f.sf(F, df_num, df_den)

    off = ~np.eye(N, dtype=bool)
    sig = np.zeros((N, N), dtype=bool)
    sig[off] = benjamini_hochberg(pvals[off], fdr)

    mk = lambda a: pd.DataFrame(a, index=data.columns, columns=data.columns)
    return mk(pvals), mk(fstat), mk(sig)


def edge_list(sig: pd.DataFrame, weight: pd.DataFrame | None = None) -> pd.DataFrame:
    """Anlamlı kenarları uzun tablo olarak verir: kaynak → hedef."""
    rows = [
        {"kaynak": j, "hedef": i, "agirlik": None if weight is None else weight.loc[i, j]}
        for i in sig.index
        for j in sig.columns
        if sig.loc[i, j]
    ]
    out = pd.DataFrame(rows, columns=["kaynak", "hedef", "agirlik"])
    return out.sort_values("agirlik", ascending=False) if weight is not None else out
