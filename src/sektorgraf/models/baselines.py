"""Merdivenin ilk basamakları: Sıfır tahmini, AR, Lasso-VAR(X)."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from .base import Model, check_no_future, lagged_design


class Zero(Model):
    """Her gün için 0 tahmin eder. Getiri tahmininde şaşırtıcı derecede güçlü bir rakip."""

    name = "Zero"

    def fit(self, train, exog=None):
        self.columns = train.columns
        return self

    def predict(self, data, dates, exog=None):
        return pd.DataFrame(0.0, index=pd.DatetimeIndex(dates), columns=self.columns)


class AR(Model):
    """Her sektör sadece kendi geçmişiyle: r_i,t = c_i + Σ_k φ_ik · r_i,t-k."""

    def __init__(self, lags: int = 1):
        self.lags = lags
        self.name = f"AR({lags})"

    def fit(self, train, exog=None):
        self.columns = train.columns
        self.coef = {}
        for c in train.columns:
            X = lagged_design(train[[c]], self.lags)
            df = pd.concat([train[c].rename("y"), X], axis=1).dropna()
            Xc = np.column_stack([np.ones(len(df)), df.drop(columns="y").to_numpy()])
            self.coef[c] = np.linalg.lstsq(Xc, df["y"].to_numpy(), rcond=None)[0]
        return self

    def predict(self, data, dates, exog=None):
        check_no_future(data, dates)
        out = {}
        for c in self.columns:
            X = lagged_design(data[[c]], self.lags).loc[dates].to_numpy()
            out[c] = self.coef[c][0] + X @ self.coef[c][1:]
        return pd.DataFrame(out, index=pd.DatetimeIndex(dates))


class LassoVAR(Model):
    """Her sektör BÜTÜN sektörlerin geçmişiyle (+ opsiyonel dış parametreler), Lasso ile seyrek.

    Sektörler arası doğrusal etkinin tahmin tarafındaki temsilcisi.
    """

    def __init__(self, lags: int = 1, alpha: float = 0.01):
        self.lags, self.alpha = lags, alpha
        self.name = f"LassoVAR({lags})"

    def fit(self, train, exog=None):
        if exog is not None:
            self.name += "X"
        self.columns = train.columns
        X = lagged_design(train, self.lags, exog)
        df = pd.concat([train, X], axis=1).dropna()
        self.features = X.columns
        self.models = {
            c: Lasso(alpha=self.alpha, max_iter=10_000).fit(df[self.features], df[c])
            for c in self.columns
        }
        return self

    def predict(self, data, dates, exog=None):
        check_no_future(data, dates)
        X = lagged_design(data, self.lags, exog).loc[dates, self.features]
        out = {c: m.predict(X) for c, m in self.models.items()}
        return pd.DataFrame(out, index=pd.DatetimeIndex(dates))

    def coefficient_matrix(self) -> pd.DataFrame:
        """1. gecikme katsayıları N×N: [i, j] = j'nin dünkü değerinin i'nin bugününe etkisi."""
        cols = [f"{c}_L1" for c in self.columns]
        rows = {i: pd.Series(m.coef_, self.features)[cols].to_numpy() for i, m in self.models.items()}
        return pd.DataFrame(rows, index=self.columns).T.set_axis(self.columns, axis=1)
