"""Adım 3: Merdivenin ilk basamakları (Zero → AR → Lasso-VAR) ve ortak değerlendirme.

Modeller eğitim döneminde eğitilir, test dönemindeki her gün için bir adım ileri tahmin yapar.
"""

import argparse

import pandas as pd

from sektorgraf.config import load_config
from sektorgraf.data.splits import split
from sektorgraf.evaluation.evaluate import compare, evaluate
from sektorgraf.models.baselines import AR, LassoVAR, Zero


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--residuals", default="ff3")
    args = ap.parse_args()

    cfg = load_config()
    m = cfg["models"]
    resid = pd.read_csv(cfg["paths"]["processed"] / f"residuals_{args.residuals}.csv",
                        index_col=0, parse_dates=True)
    parts = split(resid, cfg)
    train, test_dates = parts["train"], parts["test"].index

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
