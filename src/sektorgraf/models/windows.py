"""Derin öğrenme modelleri için kayan pencere veri hazırlığı (Graph WaveNet bunu kullanacak)."""

import numpy as np
import pandas as pd


def make_windows(data: pd.DataFrame, history: int = 20, exog: pd.DataFrame | None = None):
    """Her hedef gün t için girdi = [t-history, t-1], hedef = t.

    Dönüş:
      X     : (örnek, history, N, F)   F = 1 + dış parametre sayısı (her düğüme kopyalanır)
      y     : (örnek, N)
      dates : hedef günler
    """
    data = data.dropna()
    V = data.to_numpy()[..., None]  # (T, N, 1)
    if exog is not None:
        E = exog.reindex(data.index).ffill().to_numpy()  # (T, K)
        E = np.repeat(E[:, None, :], V.shape[1], axis=1)  # (T, N, K)
        V = np.concatenate([V, E], axis=-1)
    T = len(data)
    X = np.stack([V[t - history : t] for t in range(history, T)])
    y = data.to_numpy()[history:]
    return X.astype(np.float32), y.astype(np.float32), data.index[history:]
