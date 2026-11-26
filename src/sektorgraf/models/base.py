"""Bütün modellerin ortak arayüzü.

Kural: d günü için yapılan tahmin, SADECE d'den önceki günlerin GERÇEK değerlerini kullanır
(bir adım ileri tahmin). Model kendi tahminlerini girdi olarak kullanmaz.
"""

import numpy as np
import pandas as pd


class Model:
    name = "base"

    def fit(self, train: pd.DataFrame, exog: pd.DataFrame | None = None) -> "Model":
        raise NotImplementedError

    def predict(self, data: pd.DataFrame, dates, exog: pd.DataFrame | None = None) -> pd.DataFrame:
        """data: tahmin günlerinden öncesini de içeren tam tablo. dates: tahmin edilecek günler.

        Dönüş: index = dates, sütunlar = sektörler.
        """
        raise NotImplementedError


def lagged_design(data: pd.DataFrame, lags: int, exog: pd.DataFrame | None = None) -> pd.DataFrame:
    """Her satır t için t-1..t-lags değerlerini yan yana koyar (exog de t-1'den).

    Sütun adları: '<seri>_L<k>'. Satır t'nin özellikleri t gününden ÖNCE bilinir.
    """
    parts = [data.shift(k).add_suffix(f"_L{k}") for k in range(1, lags + 1)]
    if exog is not None:
        parts.append(exog.reindex(data.index).shift(1).add_suffix("_X1"))
    return pd.concat(parts, axis=1)


def check_no_future(data: pd.DataFrame, dates) -> None:
    dates = pd.DatetimeIndex(dates)
    if not dates.isin(data.index).all():
        raise ValueError("Tahmin günlerinin hepsi veride olmalı (gerçek değerle kıyas için).")
    if np.any(data.index.get_indexer(dates) < 1):
        raise ValueError("İlk günden önce geçmiş yok.")
