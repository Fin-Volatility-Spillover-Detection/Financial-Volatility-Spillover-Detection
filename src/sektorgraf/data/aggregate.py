"""Günlük veriyi haftalık / 2 haftalık / aylık dönemlere birleştirme.

Kurallar:
  - Getiriler BİLEŞİK birleştirilir: [(1+r1/100)·…·(1+rn/100) − 1]·100  (birim yüzde)
  - Haftalar Cuma'da biter (W-FRI).
  - 2 haftalık dönemler, 11 Ocak 1985'te biten çifte göre hizalıdır.
  - Dönem etiketi = dönemin SON İŞLEM GÜNÜ (Cuma tatilse Perşembe).
  - Aylık dönemler takvim ayıdır.
"""

import numpy as np
import pandas as pd

FREQS = ("gunluk", "haftalik", "2haftalik", "aylik")
BIWEEK_ANCHOR = pd.Timestamp("1985-01-11")  # bir 2 haftalık dönemin bittiği Cuma


def period_keys(dates: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """Her gün için ait olduğu dönemin kimliği (aynı kimlik = aynı dönem)."""
    dates = pd.DatetimeIndex(dates)
    if freq == "gunluk":
        return np.arange(len(dates))
    if freq in ("haftalik", "2haftalik"):
        week_end = dates.to_period("W-FRI").end_time.normalize()  # o haftanın Cuma'sı
        if freq == "haftalik":
            return week_end.values
        n_weeks = ((week_end - BIWEEK_ANCHOR).days // 7).to_numpy()
        # n_weeks çift → dönem sonu haftası; tek → bir sonraki haftayla birleşir
        pair_end = week_end + pd.to_timedelta((n_weeks % 2) * 7, unit="D")
        return pair_end.values
    if freq == "aylik":
        return dates.to_period("M").astype(str).to_numpy()
    raise ValueError(freq)


def _group(df: pd.DataFrame, freq: str):
    keys = period_keys(df.index, freq)
    return df.groupby(keys, sort=True), keys


def _relabel(agg: pd.DataFrame, df: pd.DataFrame, keys) -> pd.DataFrame:
    last_day = pd.Series(df.index, index=keys).groupby(level=0).max()
    agg.index = pd.DatetimeIndex(last_day.loc[agg.index].values, name="Date")
    return agg


def compound(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Yüzde getirileri dönem içinde bileşik birleştirir. Tamamen boş dönem → NaN."""
    if freq == "gunluk":
        return df.copy()
    g, keys = _group(df, freq)
    growth = np.log1p(df / 100.0)
    agg = (np.expm1(growth.groupby(keys).sum(min_count=1)) * 100.0)
    return _relabel(agg, df, keys)


def min_max(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Dönem içindeki en düşük ve en yüksek GÜNLÜK getiri (<sektör>_min, <sektör>_max)."""
    g, keys = _group(df, freq)
    mn = _relabel(g.min(), df, keys).add_suffix("_min")
    mx = _relabel(g.max(), df, keys).add_suffix("_max")
    return mn, mx


def with_min_max(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Ekipteki dosya düzeni: <sektör>, <sektör>_min, <sektör>_max, ..."""
    ret = compound(df, freq)
    if freq == "gunluk":
        return ret
    mn, mx = min_max(df, freq)
    cols = []
    for c in df.columns:
        cols += [ret[c].rename(c), mn[f"{c}_min"], mx[f"{c}_max"]]
    return pd.concat(cols, axis=1)


def ff3_aggregate(ff3: pd.DataFrame, freq: str) -> pd.DataFrame:
    """FF3 faktörlerini birleştirir.

    Mkt-RF bir FAZLA getiri olduğu için doğrudan bileşiklenmez:
      Mkt = Mkt-RF + RF (günlük) → bileşik;  RF → bileşik;  Mkt-RF(dönem) = Mkt − RF.
    SMB ve HML (sıfır maliyetli uzun-kısa portföyler) günlük değerlerinden bileşiklenir.
    n_gun: dönemdeki işlem günü sayısı.
    """
    if freq == "gunluk":
        out = ff3[["Mkt-RF", "SMB", "HML", "RF"]].copy()
        out["n_gun"] = 1
        return out
    mkt = compound((ff3["Mkt-RF"] + ff3["RF"]).to_frame("Mkt"), freq)["Mkt"]
    rf = compound(ff3[["RF"]], freq)["RF"]
    smb_hml = compound(ff3[["SMB", "HML"]], freq)
    g, keys = _group(ff3, freq)
    n = _relabel(g.size().to_frame("n_gun"), ff3, keys)["n_gun"]
    return pd.DataFrame({"Mkt-RF": mkt - rf, "SMB": smb_hml["SMB"], "HML": smb_hml["HML"], "RF": rf, "n_gun": n})


def realized_variance(daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Dönem içindeki günlük değerlerin kareleri toplamı (gerçekleşen varyans, %²) + n_gun."""
    g, keys = _group(daily, freq)
    rv = _relabel((daily ** 2).groupby(keys).sum(min_count=1), daily, keys)
    n = _relabel(daily.notna().all(axis=1).groupby(keys).size().to_frame("n_gun"), daily, keys)
    rv["n_gun"] = n["n_gun"]
    return rv
