"""Sakin / kriz ayrımı. Kriz tarihleri SADECE değerlendirmede kullanılır (sonradan bilinen bilgi)."""

import pandas as pd


def crisis_mask(dates, cfg) -> pd.Series:
    dates = pd.DatetimeIndex(dates)
    m = pd.Series(False, index=dates)
    for start, end in cfg["evaluation"]["crises"].values():
        m |= (dates >= start) & (dates <= end)
    return m


def crisis_label(dates, cfg) -> pd.Series:
    """Her gün için kriz adı ya da 'sakin'."""
    dates = pd.DatetimeIndex(dates)
    lab = pd.Series("sakin", index=dates)
    for name, (start, end) in cfg["evaluation"]["crises"].items():
        lab[(dates >= start) & (dates <= end)] = name
    return lab
