"""Adım 2: Graflar (Granger + Diebold–Yılmaz).

Girdi: data/processed/artik_<model>_<frekans>_1985.csv (config: data.frequency, data.residual_model)
Çıktı: results/<frekans>/graphs/
  dy_train.csv     : SADECE eğitim döneminden DY etki tablosu → tahmin modellerine girdi
  dy_full.csv      : bütün dönemden DY etki tablosu → ana sonuç / tarif (tahminde KULLANILMAZ)
  granger_*.csv    : eğitim dönemi Granger p-değerleri ve FDR sonrası kenar listesi
  dy_rolling.csv   : (--rolling) kayan pencereli toplam bağlantılılık ve net değerler, kriz analizi için

Örnek:  python scripts/02_build_graphs.py                      # haftalık (config)
        python scripts/02_build_graphs.py --frequency gunluk   # günlük sağlamlık
"""

import argparse

from sektorgraf.config import load_config
from sektorgraf.data.io import load_processed, results_dir
from sektorgraf.data.splits import split
from sektorgraf.graphs import dy, granger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frequency", help="gunluk | haftalik | 2haftalik | aylik (varsayılan: config)")
    ap.add_argument("--model", help="ff3 | capm (varsayılan: config)")
    ap.add_argument("--rolling", action="store_true", help="kayan pencereli DY (yavaş)")
    args = ap.parse_args()

    cfg = load_config()
    freq = args.frequency or cfg["data"]["frequency"]
    g = cfg["graphs"]
    resid = load_processed(cfg, "artik", freq, args.model)
    train = split(resid, cfg)["train"]
    out = results_dir(cfg, freq) / "graphs"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{freq}] {resid.index[0].date()} – {resid.index[-1].date()}, {resid.shape}; "
          f"eğitim {train.index[0].date()} – {train.index[-1].date()} ({len(train)} dönem)")
    kw = dict(lags=g["dy"]["lags"], horizon=g["dy"]["horizon"],
              estimator=g["dy"]["estimator"], alpha=g["dy"]["lasso_alpha"])

    # --- Granger (eğitim dönemi)
    p, F, sig = granger.granger_graph(train, g["granger"]["lags"], g["granger"]["fdr"])
    p.to_csv(out / "granger_pvals_train.csv")
    edges = granger.edge_list(sig, F)
    edges.to_csv(out / "granger_edges_train.csv", index=False)
    print(f"\nGranger: {sig.values.sum()} anlamlı kenar / {sig.size - len(sig)} çift (FDR {g['granger']['fdr']})")
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
        print(f"\nKayan DY: {len(roll)} pencere → {out / 'dy_rolling.csv'}")


if __name__ == "__main__":
    main()
