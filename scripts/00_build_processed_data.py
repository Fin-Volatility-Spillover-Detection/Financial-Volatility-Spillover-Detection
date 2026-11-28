"""Adım 0: Ham günlük veriden bütün işlenmiş veri setlerini üretir (data/processed/).

Çalıştırma:  python scripts/00_build_processed_data.py

Üretilenler (her frekans için: gunluk, haftalik, 2haftalik, aylik):
  49_sektor_<frekans>_1983.csv        Ham sektör getirileri (1983+), haftalık/2 haftalık/aylıkta min/max ile
  ff3_<frekans>_1983.csv              FF3 faktörleri aynı dönemlere birleştirilmiş (1983+)
  artik_ff3_<frekans>_1985.csv        FF3 artıkları (piyasa + SMB + HML çıkarılmış) — ANA GİRDİ
  beta_ff3_<frekans>_1985.csv         Kayan pencere FF3 betaları: <sektör>_mkt, _smb, _hml
  oynaklik_rv_ff3_<frekans>_1985.csv  Günlük FF3 artıklarından gerçekleşen varyans (haftalik/2haftalik/aylik)

--capm ile ayrıca (sağlamlık kontrolü için, varsayılan olarak üretilmez):
  artik_capm_<frekans>_1985.csv, beta_capm_<frekans>_1985.csv

Look-ahead yok: t dönemindeki betalar sadece [t−pencere, t−1] dönemlerinden tahmin edilir.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sektorgraf.config import load_config  # noqa: E402
from sektorgraf.data.aggregate import FREQS, compound, ff3_aggregate, realized_variance, with_min_max  # noqa: E402
from sektorgraf.data.io import load_aligned  # noqa: E402
from sektorgraf.data.residuals import rolling_factor_model  # noqa: E402

# Tarihler ve beta pencereleri configs/config.yaml → data_prep altında.
FMT = "%.4f"


def save(df: pd.DataFrame, path: Path) -> None:
    df.index.name = "Date"
    df.to_csv(path, float_format=FMT, date_format="%Y-%m-%d")
    print(f"  {path.name:42s} {df.index[0].date()} – {df.index[-1].date()}  {df.shape}")


def mean_corr(df: pd.DataFrame) -> float:
    c = df.corr().to_numpy()
    n = len(c)
    return float((c.sum() - n) / (n * n - n))


def beta_wide(coefs: dict, keys: dict) -> pd.DataFrame:
    parts = []
    sectors = coefs["Mkt-RF"].columns
    for s in sectors:
        for name, suffix in keys.items():
            parts.append(coefs[name][s].rename(f"{s}_{suffix}"))
    return pd.concat(parts, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capm", action="store_true", help="CAPM artıklarını ve betalarını da üret")
    args = ap.parse_args()
    models = ("ff3", "capm") if args.capm else ("ff3",)

    cfg = load_config()
    dp = cfg["data_prep"]
    RAW_START, ANALYSIS_START, WARMUP_FROM = dp["raw_start"], dp["analysis_start"], dp["warmup_from"]
    WINDOWS = {str(k): tuple(v) for k, v in dp["windows"].items()}
    out = cfg["paths"]["processed"]
    out.mkdir(parents=True, exist_ok=True)

    sectors_d, ff3_d = load_aligned(cfg)
    sectors_d, ff3_d = sectors_d.loc[WARMUP_FROM:], ff3_d.loc[WARMUP_FROM:]
    print(f"Ham günlük veri: {sectors_d.index[0].date()} – {sectors_d.index[-1].date()}, "
          f"{sectors_d.shape}, eksik değer: {int(sectors_d.isna().sum().sum())}")

    summary, daily_resid_ff3 = [], None
    for freq in FREQS:
        window, min_obs = WINDOWS[freq]
        print(f"\n[{freq}]  beta penceresi = {window} dönem")
        sec = compound(sectors_d, freq)
        ff = ff3_aggregate(ff3_d, freq)

        save(with_min_max(sectors_d, freq).loc[RAW_START:], out / f"49_sektor_{freq}_1983.csv")
        save(ff.loc[RAW_START:], out / f"ff3_{freq}_1983.csv")

        row = {"frekans": freq, "pencere": window, "ham_korelasyon": mean_corr(sec.loc[ANALYSIS_START:])}
        for model in models:
            resid, coefs = rolling_factor_model(sec, ff, model, window, min_obs)
            resid = resid.loc[ANALYSIS_START:]
            if resid.isna().any().any():
                raise RuntimeError(f"{model}/{freq}: 1985 sonrası artıklarda NaN var")
            save(resid, out / f"artik_{model}_{freq}_1985.csv")
            keys = {"Mkt-RF": "mkt"} if model == "capm" else {"Mkt-RF": "mkt", "SMB": "smb", "HML": "hml"}
            betas = beta_wide({k: v.loc[ANALYSIS_START:] for k, v in coefs.items()}, keys)
            save(betas, out / f"beta_{model}_{freq}_1985.csv")
            row[f"{model}_korelasyon"] = mean_corr(resid)
            row[f"{model}_medyan_piyasa_betasi"] = float(coefs["Mkt-RF"].loc[ANALYSIS_START:].median().median())
            if model == "ff3" and freq == "gunluk":
                daily_resid_ff3 = rolling_factor_model(sectors_d, ff3_d, "ff3", window, min_obs)[0]
        summary.append(row)

    print("\n[oynaklık] günlük FF3 artıklarının dönem içi kareleri toplamı")
    for freq in ("haftalik", "2haftalik", "aylik"):
        rv = realized_variance(daily_resid_ff3.dropna(how="all"), freq).loc[ANALYSIS_START:]
        save(rv, out / f"oynaklik_rv_ff3_{freq}_1985.csv")

    print("\nÖzet (1985+, sektörler arası ortalama korelasyon):")
    print(pd.DataFrame(summary).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
