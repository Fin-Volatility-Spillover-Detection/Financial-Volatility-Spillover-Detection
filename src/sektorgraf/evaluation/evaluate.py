"""ORTAK DEĞERLENDİRME. Bütün modeller sadece bunu kullanır.

    preds = model.predict(artiklar, test_gunleri)
    evaluate(preds, artiklar, cfg)               # sonuçlar results/tables/metrics.csv'ye yazılır
    compare("GWN[dy]", "GWN[none]", artiklar, cfg)  # Diebold–Mariano

Tahminler results/predictions/<model>.csv olarak saklanır; karşılaştırmalar oradan okunur.
"""

import re

import pandas as pd

from . import metrics as M
from .periods import crisis_mask


def _safe(name: str) -> str:
    return re.sub(r"[^\w\-\[\]\(\)]", "_", name)


def _scores(pred, actual) -> dict:
    return {
        "rmse": M.rmse(pred, actual),
        "mae": M.mae(pred, actual),
        "r2_oos": M.r2_oos(pred, actual),
        "yon_isabeti": M.direction_accuracy(pred, actual),
        "n_donem": len(pred),
    }


def evaluate(preds: pd.DataFrame, actual_all: pd.DataFrame, cfg, name: str | None = None,
             save: bool = True) -> pd.DataFrame:
    name = name or getattr(preds, "name", None) or "model"
    test_start, test_end = cfg["split"]["test"]
    preds = preds.loc[test_start:test_end]
    actual = actual_all.loc[preds.index, preds.columns]
    if preds.isna().any().any():
        raise ValueError(f"{name}: tahminlerde NaN var")

    crisis = crisis_mask(preds.index, cfg)
    rows = []
    for period, mask in [("tum", slice(None)), ("sakin", ~crisis.values), ("kriz", crisis.values)]:
        if period != "tum" and mask.sum() == 0:
            continue
        rows.append({"model": name, "donem": period, **_scores(preds[mask], actual[mask])})
    table = pd.DataFrame(rows)

    if save:
        out = cfg["paths"]["results"]
        (out / "predictions").mkdir(parents=True, exist_ok=True)
        (out / "tables").mkdir(parents=True, exist_ok=True)
        preds.to_csv(out / "predictions" / f"{_safe(name)}.csv")
        path = out / "tables" / "metrics.csv"
        old = pd.read_csv(path) if path.exists() else pd.DataFrame()
        if len(old):
            old = old[old["model"] != name]
        pd.concat([old, table]).to_csv(path, index=False)

        per_sector = ((preds - actual) ** 2).mean() ** 0.5
        per_sector.rename(name).to_frame().to_csv(out / "tables" / f"rmse_sektor_{_safe(name)}.csv")
    return table


def load_predictions(name: str, cfg) -> pd.DataFrame:
    path = cfg["paths"]["results"] / "predictions" / f"{_safe(name)}.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True)


def compare(name_a: str, name_b: str, actual_all: pd.DataFrame, cfg) -> pd.DataFrame:
    """A, B'den daha mı iyi? Tüm / sakin / kriz için Diebold–Mariano."""
    a, b = load_predictions(name_a, cfg), load_predictions(name_b, cfg)
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx, a.columns]
    actual = actual_all.loc[idx, a.columns]
    crisis = crisis_mask(idx, cfg).values
    rows = []
    for period, m in [("tum", slice(None)), ("sakin", ~crisis), ("kriz", crisis)]:
        if period != "tum" and m.sum() < 30:
            continue
        rows.append({"A": name_a, "B": name_b, "donem": period,
                     **M.diebold_mariano(a[m], b[m], actual[m])})
    return pd.DataFrame(rows)
