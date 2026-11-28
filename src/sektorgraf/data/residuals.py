"""CAPM / FF3 artıkları (piyasadan arındırılmış getiri).

artık_t = (sektör_t − RF_t) − (α + β · faktörler_t)

Look-ahead yok: t günündeki α ve β, SADECE [t−window, t−1] aralığından tahmin edilir.
"""

import numpy as np
import pandas as pd

FACTORS = {"capm": ["Mkt-RF"], "ff3": ["Mkt-RF", "SMB", "HML"]}


def rolling_factor_model(
    sectors: pd.DataFrame,
    ff3: pd.DataFrame,
    model: str = "ff3",
    window: int = 252,
    min_obs: int = 200,
    refit_every: int = 1,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Kayan pencereli faktör modeli: artıklar + BÜTÜN katsayılar.

    Dönüş: (artıklar, {"alpha": ..., "Mkt-RF": ..., "SMB": ..., "HML": ...})
    Her katsayı tablosu sektörlerle aynı şekilde (dönem × sektör). t satırındaki katsayı,
    t'nin artığını hesaplamakta kullanılan, [t−window, t−1] penceresinden tahmin edilmiş değerdir.
    """
    cols = FACTORS[model]
    excess = sectors.sub(ff3["RF"], axis=0).to_numpy()
    X_all = np.column_stack([np.ones(len(ff3)), ff3[cols].to_numpy()])
    T, N = excess.shape
    K = X_all.shape[1]

    resid = np.full((T, N), np.nan)
    coefs = np.full((K, T, N), np.nan)
    coef = None

    for t in range(window, T):
        if coef is None or (t - window) % refit_every == 0:
            Y = excess[t - window : t]
            X = X_all[t - window : t]
            coef = np.full((K, N), np.nan)
            ok = (~np.isnan(Y)).sum(axis=0) >= min_obs
            full = ok & ~np.isnan(Y).any(axis=0)
            if full.any():
                coef[:, full] = np.linalg.lstsq(X, Y[:, full], rcond=None)[0]
            for j in np.where(ok & ~full)[0]:
                m = ~np.isnan(Y[:, j])
                coef[:, j] = np.linalg.lstsq(X[m], Y[m, j], rcond=None)[0]
        resid[t] = excess[t] - X_all[t] @ coef
        coefs[:, t] = coef

    mk = lambda a: pd.DataFrame(a, index=sectors.index, columns=sectors.columns)
    names = ["alpha"] + cols
    return mk(resid), {n: mk(coefs[k]) for k, n in enumerate(names)}


def rolling_residuals(
    sectors: pd.DataFrame,
    ff3: pd.DataFrame,
    model: str = "ff3",
    window: int = 252,
    min_obs: int = 200,
    refit_every: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kayan pencereli artıkları ve PİYASA betalarını döndürür (geriye uyumlu kısa yol).

    Dönüş: (artıklar, piyasa_betaları). İkisi de sektörlerle aynı şekilde.
    Bütün katsayılar için rolling_factor_model kullanın.
    """
    resid, coefs = rolling_factor_model(sectors, ff3, model, window, min_obs, refit_every)
    return resid, coefs["Mkt-RF"]
