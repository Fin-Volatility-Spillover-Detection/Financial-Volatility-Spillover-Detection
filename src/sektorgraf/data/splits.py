"""Eğitim / doğrulama / test ayrımı. Tarihler sadece configs/config.yaml'da tanımlıdır."""

import pandas as pd


def split(df: pd.DataFrame, cfg) -> dict[str, pd.DataFrame]:
    s = cfg["split"]
    return {name: df.loc[s[name][0] : s[name][1]] for name in ("train", "val", "test")}


def history_until(df: pd.DataFrame, cfg, part: str) -> pd.DataFrame:
    """Bir bölümün sonuna kadar olan bütün geçmiş (ör. 'val' → train + val).

    Test için son modeli eğitirken train+val birlikte kullanılabilir.
    """
    return df.loc[: cfg["split"][part][1]]
