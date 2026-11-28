"""Adım 3: Merdivenin ilk basamakları (Zero → AR → Lasso-VAR) ve ortak değerlendirme.

Girdi: data/processed/artik_<model>_<frekans>_1985.csv (config: data.frequency, data.residual_model)
Modeller eğitim döneminde eğitilir, test dönemindeki her dönem için bir adım ileri tahmin yapar.
Sonuçlar: results/<frekans>/tables/metrics.csv, results/<frekans>/predictions/

Örnek:  python scripts/03_run_baselines.py                      # haftalık (config)
        python scripts/03_run_baselines.py --frequency gunluk   # günlük sağlamlık
"""

import argparse

import pandas as pd

from sektorgraf.config import load_config
from sektorgraf.data.io import load_processed, results_dir
from sektorgraf.data.splits import split
from sektorgraf.evaluation.evaluate import compare, evaluate
from sektorgraf.models.baselines import AR, LassoVAR, Zero


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frequency", help="gunluk | haftalik | 2haftalik | aylik (varsayılan: config)")
    ap.add_argument("--model", help="ff3 | capm (varsayılan: config)")
    args = ap.parse_args()

    cfg = load_config()
    freq = args.frequency or cfg["data"]["frequency"]
    cfg["paths"]["results"] = results_dir(cfg, freq)   # evaluate() buraya yazar
    m = cfg["models"]
    resid = load_processed(cfg, "artik", freq, args.model)
    parts = split(resid, cfg)
    train, test_dates = parts["train"], parts["test"].index
    print(f"[{freq}] eğitim {train.index[0].date()} – {train.index[-1].date()} ({len(train)}), "
          f"test {test_dates[0].date()} – {test_dates[-1].date()} ({len(test_dates)})\n")

    models = [Zero(), AR(m["ar"]["lags"]), LassoVAR(m["lasso_var"]["lags"], m["lasso_var"]["alpha"])]
    tables = []
    for model in models:
        model.fit(train)
        preds = model.predict(resid, test_dates)
        tables.append(evaluate(preds, resid, cfg, name=model.name))
    print(pd.concat(tables).round(4).to_string(index=False))

    print("\nDiebold–Mariano (p_a_better < 0.05 → A anlamlı derecede iyi):")
    for a, b in [(models[1].name, "Zero"), (models[2].name, models[1].name)]:
        print(compare(a, b, resid, cfg)[["A", "B", "donem", "stat", "p_a_better"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
