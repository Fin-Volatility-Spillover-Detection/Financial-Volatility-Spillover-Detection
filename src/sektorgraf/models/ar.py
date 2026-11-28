"""AR: her sektör sadece kendi geçmişiyle tahmin edilir (BASLANGIC.md 6.3, is_bolumu.md 2.3).

    r[i, t] = c_i + φ_i1 · r[i, t−1] + … + φ_ip · r[i, t−p] + ε

49 sektörün her biri için ayrı bir en küçük kareler regresyonu. Katsayılar eğitim döneminden
tahmin edilir ve sonra sabit kalır. Tahmin bir adım ileridir: t tahmini sadece t−1 ve öncesinin
gerçek değerleriyle yapılır.
"""

import numpy as np
import pandas as pd


def _lags(df: pd.DataFrame, p: int) -> list[pd.DataFrame]:
    """[df.shift(1), …, df.shift(p)]: k. eleman, her satırda k dönem önceki değer."""
    return [df.shift(k) for k in range(1, p + 1)]


def fit_ar(train: pd.DataFrame, p: int) -> pd.DataFrame:
    """Eğitim tablosundaki her sektör için AR(p) katsayıları.

    Gecikmeler sadece eğitim tablosunun içinden alınır; ilk p satırın gecikmesi eksik olduğu
    için bu satırlar regresyona girmez.
    Dönüş: satırlar ["c", "lag1", …, "lagp"], sütunlar sektörler.
    """
    lags = _lags(train, p)
    rows = slice(p, None)
    coefs = {}
    for col in train.columns:
        X = np.column_stack([np.ones(len(train) - p)] + [lag[col].values[rows] for lag in lags])
        y = train[col].values[rows]
        coefs[col], *_ = np.linalg.lstsq(X, y, rcond=None)
    return pd.DataFrame(coefs, index=["c"] + [f"lag{k}" for k in range(1, p + 1)])[train.columns]


def predict_ar(history: pd.DataFrame, coefs: pd.DataFrame, dates: pd.Index) -> pd.DataFrame:
    """Verilen tarihler için bir adım ileri AR tahminleri.

    history: tahmin tarihlerinden önceki en az p dönemi de içeren gerçek değer tablosu
    (ör. doğrulama için eğitim + doğrulama). t tarihinin tahmini history'deki t−1 … t−p
    satırlarından hesaplanır; t'nin kendisi ve sonrası kullanılmaz.
    """
    p = len(coefs) - 1
    lags = _lags(history, p)
    pred = pd.DataFrame(np.broadcast_to(coefs.loc["c"].values, (len(history), history.shape[1])),
                        index=history.index, columns=history.columns)
    for k, lag in enumerate(lags, start=1):
        pred = pred + lag * coefs.loc[f"lag{k}"]
    pred = pred.loc[dates]
    if pred.isna().any().any():
        raise ValueError("Bazı tarihler için yeterli geçmiş yok (history en az p dönem önceden başlamalı).")
    return pred
