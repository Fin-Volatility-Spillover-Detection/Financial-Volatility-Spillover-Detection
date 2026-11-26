"""Diebold–Yılmaz bağlantılılık (connectedness) — projenin ana etki tablosu.

1. VAR(p) tahmini (49 sektör için Lasso ile seyrek; Demirer, Diebold, Liu & Yılmaz 2018).
2. Genelleştirilmiş tahmin hatası varyans ayrıştırması (GFEVD, Pesaran–Shin 1998):
     θ_ij(H) = σ_jj⁻¹ Σ_h (e_iᵀ Ψ_h Σ e_j)² / Σ_h (e_iᵀ Ψ_h Σ Ψ_hᵀ e_i)
   "i'nin H günlük tahmin hatası varyansının yüzde kaçı j'den geliyor?"
3. Satırlar 100'e normalize edilir (Diebold–Yılmaz 2012).

Not: GFEVD hata kovaryansı Σ üzerinden AYNI GÜN yayılmayı da içerir; Granger içermez.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso


def fit_var(Y: np.ndarray, lags: int, estimator: str = "lasso", alpha: float = 0.01):
    """Dönüş: (A, Sigma). A şekli (lags, N, N); A[l][i, j] = j'nin l+1 gecikmeli etkisi i'ye."""
    T, N = Y.shape
    X = np.hstack([Y[lags - l - 1 : T - l - 1] for l in range(lags)])  # (T-p, N*p)
    y = Y[lags:]
    if estimator == "ols":
        Xc = np.hstack([np.ones((len(X), 1)), X])
        B = np.linalg.lstsq(Xc, y, rcond=None)[0]
        coef, resid = B[1:].T, y - Xc @ B
    elif estimator == "lasso":
        coef = np.zeros((N, N * lags))
        resid = np.zeros_like(y)
        for i in range(N):
            m = Lasso(alpha=alpha, max_iter=10_000).fit(X, y[:, i])
            coef[i], resid[:, i] = m.coef_, y[:, i] - m.predict(X)
    else:
        raise ValueError(estimator)
    A = coef.reshape(N, lags, N).transpose(1, 0, 2)
    Sigma = np.cov(resid, rowvar=False)
    return A, Sigma


def ma_coefficients(A: np.ndarray, horizon: int) -> np.ndarray:
    """VAR → hareketli ortalama gösterimi: Ψ_0 = I, Ψ_h = Σ_l A_l Ψ_{h-l}."""
    p, N, _ = A.shape
    Psi = np.zeros((horizon, N, N))
    Psi[0] = np.eye(N)
    for h in range(1, horizon):
        for l in range(1, min(h, p) + 1):
            Psi[h] += A[l - 1] @ Psi[h - l]
    return Psi


def gfevd(A: np.ndarray, Sigma: np.ndarray, horizon: int) -> np.ndarray:
    """Satırları 100'e normalize edilmiş θ (N×N)."""
    Psi = ma_coefficients(A, horizon)
    num = sum((P @ Sigma) ** 2 for P in Psi) / np.diag(Sigma)[None, :]
    den = sum(np.diag(P @ Sigma @ P.T) for P in Psi)
    theta = num / den[:, None]
    return 100 * theta / theta.sum(axis=1, keepdims=True)


def connectedness(theta: pd.DataFrame) -> dict:
    """Toplam, yönlü (to/from) ve net bağlantılılık. theta satırları 100'e normalize olmalı."""
    off = theta.to_numpy().copy()
    np.fill_diagonal(off, 0.0)
    frm = off.sum(axis=1)  # i'nin diğerlerinden aldığı
    to = off.sum(axis=0)   # j'nin diğerlerine verdiği
    idx = theta.index
    return {
        "total": off.sum() / len(idx),
        "from": pd.Series(frm, idx),
        "to": pd.Series(to, idx),
        "net": pd.Series(to - frm, idx),  # + net şok verici, − net şok alıcı
    }


def dy_table(data: pd.DataFrame, lags=1, horizon=10, estimator="lasso", alpha=0.01) -> pd.DataFrame:
    """Ana fonksiyon: N×N etki tablosu (DataFrame), [i, j] = j'nin i'ye etkisi (%)."""
    data = data.dropna()
    A, Sigma = fit_var(data.to_numpy(), lags, estimator, alpha)
    theta = gfevd(A, Sigma, horizon)
    return pd.DataFrame(theta, index=data.columns, columns=data.columns)


def rolling_total(data: pd.DataFrame, window=250, step=5, **kw) -> pd.DataFrame:
    """Kayan pencereli toplam bağlantılılık + net değerler (kapak grafiği adayı).

    Her tarih, kendisiyle biten pencereden hesaplanır (sadece geçmiş).
    """
    data = data.dropna()
    rows = []
    for end in range(window, len(data) + 1, step):
        chunk = data.iloc[end - window : end]
        c = connectedness(dy_table(chunk, **kw))
        rows.append({"date": chunk.index[-1], "total": c["total"], **c["net"].add_prefix("net_")})
    return pd.DataFrame(rows).set_index("date")
