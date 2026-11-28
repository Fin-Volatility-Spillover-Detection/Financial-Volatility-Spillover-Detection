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


# --- İşlenmiş veri (scripts/00_build_processed_data.py çıktıları) -------------------------

_PROCESSED = {
    "artik": "artik_{model}_{freq}_1985.csv",          # ana girdi: faktör modeli artıkları
    "beta": "beta_{model}_{freq}_1985.csv",
    "rv": "oynaklik_rv_ff3_{freq}_1985.csv",           # gerçekleşen varyans (+ n_gun)
    "ham": "49_sektor_{freq}_1983.csv",                # ham getiriler (min/max sütunları atılır)
    "faktor": "ff3_{freq}_1983.csv",                   # Mkt-RF, SMB, HML, RF, n_gun
}


def load_processed(cfg, kind: str = "artik", frequency: str | None = None,
                   model: str | None = None) -> pd.DataFrame:
    """data/processed/ altındaki bir tabloyu okur.

    frequency / model verilmezse config'teki data.frequency / data.residual_model kullanılır.
    'ham' ve 'faktor' analiz başlangıcından (data_prep.analysis_start) itibaren kesilir, böylece
    bütün tablolar aynı tarihlerle hizalı olur. data.drop_last_partial ise günlük dışındaki
    frekanslarda son (kısmi) satır atılır.
    """
    d = cfg["data"]
    freq = frequency or d["frequency"]
    model = model or d["residual_model"]
    name = _PROCESSED[kind].format(freq=freq, model=model)
    path = cfg["paths"]["processed"] / name
    if not path.exists():
        hint = " --capm" if model == "capm" else ""
        raise FileNotFoundError(f"{path} yok. Önce: python scripts/00_build_processed_data.py{hint}")
    df = _read(path).astype(float)
    if kind == "ham":
        df = df[[c for c in df.columns if not c.endswith(("_min", "_max"))]]
    if kind in ("ham", "faktor"):
        df = df.loc[cfg["data_prep"]["analysis_start"]:]
    if d.get("drop_last_partial", True) and freq != "gunluk":
        df = df.iloc[:-1]
    return df


def results_dir(cfg, frequency: str | None = None):
    """Sonuçların yazılacağı klasör: results/<frekans>/ (frekanslar birbirinin üstüne yazmasın)."""
    return cfg["paths"]["results"] / (frequency or cfg["data"]["frequency"])
