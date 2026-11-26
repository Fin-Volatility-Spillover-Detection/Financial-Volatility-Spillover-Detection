"""Adım 1: Ham getiriler → kayan pencereli FF3 (ve CAPM) artıkları.

Çıktı: data/processed/residuals_<model>.csv, betas_<model>.csv
Sonraki BÜTÜN adımlar bu artık tablosunu girdi olarak alır.
"""

import argparse

from sektorgraf.config import load_config
from sektorgraf.data.io import load_aligned
from sektorgraf.data.residuals import rolling_residuals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ff3", "capm", "both"], default="both")
    args = ap.parse_args()

    cfg = load_config()
    rc = cfg["residuals"]
    sectors, ff3 = load_aligned(cfg)
    sectors, ff3 = sectors.loc[rc["start"]:], ff3.loc[rc["start"]:]

    out = cfg["paths"]["processed"]
    out.mkdir(parents=True, exist_ok=True)
    for model in (["ff3", "capm"] if args.model == "both" else [args.model]):
        resid, betas = rolling_residuals(sectors, ff3, model, rc["window"], rc["min_obs"], rc["refit_every"])
        resid, betas = resid.dropna(how="all"), betas.dropna(how="all")
        resid.to_csv(out / f"residuals_{model}.csv")
        betas.to_csv(out / f"betas_{model}.csv")
        ham = sectors.loc[resid.index]
        print(f"[{model}] {resid.index[0].date()} – {resid.index[-1].date()}, {resid.shape}")
        print(f"  ort. çapraz korelasyon: ham {_mean_corr(ham):.2f} → artık {_mean_corr(resid):.2f}")


def _mean_corr(df):
    c = df.corr().to_numpy()
    n = len(c)
    return (c.sum() - n) / (n * n - n)


if __name__ == "__main__":
    main()
