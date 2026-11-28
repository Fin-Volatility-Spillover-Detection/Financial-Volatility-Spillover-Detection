"""Ortak veri okuma ve zaman ayrımı. Bütün ekip ayrımı buradan alır.

    from sektorgraf.config import load_config
    from sektorgraf.data.splits import load_processed, split

    cfg = load_config()
    artik = load_processed(cfg, "artik_ff3_haftalik_1985.csv")   # son kısmi dönem atılmış
    parts = split(artik, cfg)                                     # {"train", "val", "test"}
"""

import pandas as pd

PARTS = ("train", "val", "test")


def load_processed(cfg, filename: str, drop_last_partial: bool = True) -> pd.DataFrame:
    """data/processed/ altındaki bir CSV'yi okur.

    Haftalık, 2 haftalık ve aylık dosyalarda son satır (2026-09-30) kısmi bir dönemdir;
    drop_last_partial=True ise atılır. Günlük dosyalarda son satır tam gündür, atılmaz.
    """
    path = cfg["paths"]["processed"] / filename
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    if drop_last_partial and "_gunluk_" not in filename:
        df = df.iloc[:-1]
    return df


def split(df: pd.DataFrame, cfg) -> dict[str, pd.DataFrame]:
    """Tabloyu config'teki tarihlere göre eğitim / doğrulama / test olarak böler."""
    s = cfg["split"]
    return {p: df.loc[s[p][0] : s[p][1]] for p in PARTS}
