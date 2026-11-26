"""CAPM / FF3 artıkları (piyasadan arındırılmış getiri).

artık_t = (sektör_t − RF_t) − (α + β · faktörler_t)

Look-ahead yok: t günündeki α ve β, SADECE [t−window, t−1] aralığından tahmin edilir.
"""

import numpy as np
import pandas as pd

FACTORS = {"capm": ["Mkt-RF"], "ff3": ["Mkt-RF", "SMB", "HML"]}


def rolling_residuals(
    sectors: pd.DataFrame,
    ff3: pd.DataFrame,
    model: str = "ff3",
    window: int = 252,
    min_obs: int = 200,
    refit_every: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kayan pencereli artıkları ve betaları döndürür.

    Dönüş: (artıklar, piyasa_betaları). İkisi de sektörlerle aynı şekilde (gün × sektör).
    """
    cols = FACTORS[model]
    excess = sectors.sub(ff3["RF"], axis=0).to_numpy()
    X_all = np.column_stack([np.ones(len(ff3)), ff3[cols].to_numpy()])
    T, N = excess.shape

    resid = np.full((T, N), np.nan)
    betas = np.full((T, N), np.nan)
    coef = None

    for t in range(window, T):
        if coef is None or (t - window) % refit_every == 0:
            Y = excess[t - window : t]
            X = X_all[t - window : t]
            coef = np.full((X.shape[1], N), np.nan)
            ok = (~np.isnan(Y)).sum(axis=0) >= min_obs
            full = ok & ~np.isnan(Y).any(axis=0)
            if full.any():  # eksiksiz sektörler: tek seferde
                coef[:, full] = np.linalg.lstsq(X, Y[:, full], rcond=None)[0]
            for j in np.where(ok & ~full)[0]:  # eksikli sektörler: tek tek
                m = ~np.isnan(Y[:, j])
                coef[:, j] = np.linalg.lstsq(X[m], Y[m, j], rcond=None)[0]
        resid[t] = excess[t] - X_all[t] @ coef
        betas[t] = coef[1]

    resid = pd.DataFrame(resid, index=sectors.index, columns=sectors.columns)
    betas = pd.DataFrame(betas, index=sectors.index, columns=sectors.columns)
    return resid, betas
