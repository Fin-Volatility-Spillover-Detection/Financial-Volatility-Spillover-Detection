"""Zero: her dönem, her sektör için tahmin 0 (BASLANGIC.md, bölüm 6.2). En alt çıta."""

import pandas as pd


def predict_zero(actual: pd.DataFrame) -> pd.DataFrame:
    """Gerçek değer tablosuyla aynı tarih ve sütunlarda, bütün değerleri 0 olan tahmin tablosu."""
    return pd.DataFrame(0.0, index=actual.index, columns=actual.columns)
