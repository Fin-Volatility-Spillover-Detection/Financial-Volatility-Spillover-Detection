"""Adım 2: Graflar.

  dy_train.csv     : SADECE eğitim döneminden DY etki tablosu → tahmin modellerine girdi
  dy_full.csv      : bütün dönemden DY etki tablosu → ana sonuç / tarif (tahminde KULLANILMAZ)
  granger_*.csv    : eğitim dönemi Granger p-değerleri ve FDR sonrası kenar listesi
  dy_rolling.csv   : (--rolling) kayan pencereli toplam bağlantılılık, kriz analizi için
"""

import argparse

import pandas as pd

from sektorgraf.config import load_config
from sektorgraf.data.splits import split
from sektorgraf.graphs import dy, granger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--residuals", default="ff3")
    ap.add_argument("--rolling", action="store_true", help="kayan pencereli DY (yavaş)")
    args = ap.parse_args()

    cfg = load_config()
    g = cfg["graphs"]
    resid = pd.read_csv(cfg["paths"]["processed"] / f"residuals_{args.residuals}.csv",
                        index_col=0, parse_dates=True)
    train = split(resid, cfg)["train"]
    out = cfg["paths"]["results"] / "graphs"
    out.mkdir(parents=True, exist_ok=True)
    kw = dict(lags=g["dy"]["lags"], horizon=g["dy"]["horizon"],
              estimator=g["dy"]["estimator"], alpha=g["dy"]["lasso_alpha"])

    # --- Granger (eğitim dönemi)
    p, F, sig = granger.granger_graph(train, g["granger"]["lags"], g["granger"]["fdr"])
    p.to_csv(out / "granger_pvals_train.csv")
    edges = granger.edge_list(sig, F)
    edges.to_csv(out / "granger_edges_train.csv", index=False)
    print(f"Granger: {sig.values.sum()} anlamlı kenar / {sig.size - len(sig)} çift (FDR {g['granger']['fdr']})")
    print(edges.head(10).to_string(index=False))

    # --- Diebold–Yılmaz
    for label, data in [("train", train), ("full", resid)]:
        theta = dy.dy_table(data, **kw)
        theta.to_csv(out / f"dy_{label}.csv")
        c = dy.connectedness(theta)
        print(f"\nDY [{label}] toplam bağlantılılık: {c['total']:.1f}%")
        print("  en çok şok veren (net):", ", ".join(c["net"].nlargest(5).index))
        print("  en çok şok alan  (net):", ", ".join(c["net"].nsmallest(5).index))

    if args.rolling:
        roll = dy.rolling_total(resid, g["dy"]["rolling_window"], g["dy"]["rolling_step"], **kw)
        roll.to_csv(out / "dy_rolling.csv")
        print(f"\nKayan DY: {len(roll)} pencere → dy_rolling.csv")


if __name__ == "__main__":
    main()
