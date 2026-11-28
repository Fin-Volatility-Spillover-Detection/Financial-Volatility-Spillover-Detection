"""Ham verileri okur. Bütün tablolar: satır = gün (DatetimeIndex), sütun = seri, birim = yüzde."""

import pandas as pd


def _read(path) -> pd.DataFrame:
    # utf-8-sig: sektör dosyasının başındaki gizli BOM karakterini temizler
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["Date"], index_col="Date")
    df = df.sort_index()
    if df.index.has_duplicates:
        raise ValueError(f"{path}: tekrar eden tarih var")
    return df


def load_sectors(cfg) -> pd.DataFrame:
    """49 sektörün günlük getirileri. Erken dönemde henüz var olmayan sektörler NaN."""
    return _read(cfg["paths"]["sectors"]).astype(float)


def load_ff3(cfg) -> pd.DataFrame:
    """Fama-French faktörleri: Mkt-RF, SMB, HML, RF."""
    return _read(cfg["paths"]["ff3"]).astype(float)


def load_aligned(cfg):
    """Sektörleri ve faktörleri aynı günlere hizalar."""
    sectors, ff3 = load_sectors(cfg), load_ff3(cfg)
    idx = sectors.index.intersection(ff3.index)
    return sectors.loc[idx], ff3.loc[idx]
