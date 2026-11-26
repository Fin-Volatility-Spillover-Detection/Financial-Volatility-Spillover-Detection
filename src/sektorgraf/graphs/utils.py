"""Graf yardımcıları: modele verilecek komşuluk matrisleri ve kontrol (rastgele) graf."""

import numpy as np
import pandas as pd


def off_diagonal(W: pd.DataFrame) -> pd.DataFrame:
    A = W.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(A, 0.0)
    return pd.DataFrame(A, index=W.index, columns=W.columns)


def top_k_per_row(W: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Her sektör için en güçlü k etkileyeni tutar, gerisini sıfırlar (GNN'e verilecek seyrek graf)."""
    W = off_diagonal(W)
    out = pd.DataFrame(0.0, index=W.index, columns=W.columns)
    for i in W.index:
        top = W.loc[i].nlargest(k).index
        out.loc[i, top] = W.loc[i, top]
    return out


def random_like(W: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Kontrol grafı: W ile AYNI kenar sayısı ve AYNI ağırlıklar, ama rastgele yerlerde.

    "Herhangi bir graf mı işe yarıyor, yoksa bizim bulduğumuz mu?" testi için.
    """
    rng = np.random.default_rng(seed)
    W = off_diagonal(W)
    N = len(W)
    off = ~np.eye(N, dtype=bool)
    weights = W.to_numpy()[off]
    nz = weights[weights != 0]
    slots = rng.choice(np.flatnonzero(off), size=len(nz), replace=False)
    R = np.zeros(N * N)
    R[slots] = rng.permutation(nz)
    return pd.DataFrame(R.reshape(N, N), index=W.index, columns=W.columns)


def similarity(W1: pd.DataFrame, W2: pd.DataFrame, k: int = 50) -> dict:
    """İki graf ne kadar benzer? (ör. DY ile Graph WaveNet'in öğrendiği graf)

    spearman: bütün kenar ağırlıklarının sıra korelasyonu
    top_k_overlap: en güçlü k kenarın kaçı ortak (0–1)
    """
    from scipy.stats import spearmanr

    off = ~np.eye(len(W1), dtype=bool)
    a, b = W1.to_numpy()[off], W2.loc[W1.index, W1.columns].to_numpy()[off]
    top_a, top_b = set(np.argsort(a)[-k:]), set(np.argsort(b)[-k:])
    return {"spearman": spearmanr(a, b).statistic, "top_k_overlap": len(top_a & top_b) / k}
