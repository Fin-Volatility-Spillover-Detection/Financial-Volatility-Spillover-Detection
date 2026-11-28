"""Bütün modellerin tahmin dosyalarını okuyup ortak karşılaştırma tablosunu yazar (docs/is_bolumu.md, bölüm 6).

    python scripts/evaluate.py                    # haftalık, test dönemi
    python scripts/evaluate.py --freq gunluk      # günlük sağlamlık kontrolü
    python scripts/evaluate.py --part val         # doğrulama dönemi (sadece bilgi amaçlı)

Okur:   results/<freq>/predictions/<model>_<part>.csv
Yazar:  results/<freq>/tables/karsilastirma[_val].csv   model başına RMSE, R² (Zero'ya göre), yön isabeti
        results/<freq>/tables/dm_testleri[_val].csv     ikili Diebold–Mariano testleri
        results/<freq>/tables/sektor_rmse[_val].csv     sektör × model RMSE
Henüz dosyası olmayan modeller atlanır.
"""

import argparse

import pandas as pd

from sektorgraf.config import ROOT, load_config
from sektorgraf.data.splits import load_processed, split
from sektorgraf.evaluation.metrics import diebold_mariano, direction_accuracy, oos_r2, rmse, rmse_by_sector

MODELS = ["zero", "ar", "var", "lstm"]
DM_PAIRS = [("ar", "zero"), ("var", "ar"), ("var", "zero"), ("lstm", "ar"), ("lstm", "zero")]


def load_predictions(path, actual: pd.DataFrame) -> pd.DataFrame:
    pred = pd.read_csv(path, index_col="Date", parse_dates=True)
    if pred.shape != actual.shape:
        raise ValueError(f"{path.name}: boyut {pred.shape}, beklenen {actual.shape}.")
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default="haftalik", choices=["haftalik", "gunluk"])
    ap.add_argument("--part", default="test", choices=["val", "test"])
    args = ap.parse_args()

    cfg = load_config()
    artik = load_processed(cfg, f"artik_ff3_{args.freq}_1985.csv")
    actual = split(artik, cfg)[args.part]

    pred_dir = ROOT / "results" / args.freq / "predictions"
    preds = {}
    for m in MODELS:
        path = pred_dir / f"{m}_{args.part}.csv"
        if path.exists():
            preds[m] = load_predictions(path, actual)
    if not preds:
        raise SystemExit(f"{pred_dir} altında {args.part} tahmin dosyası yok.")

    table = pd.DataFrame(
        {
            m: {"RMSE": rmse(p, actual), "R2_zero": oos_r2(p, actual), "yon_isabeti": direction_accuracy(p, actual)}
            for m, p in preds.items()
        }
    ).T.rename_axis("model")

    dm_rows = []
    for a, b in DM_PAIRS:
        if a in preds and b in preds:
            dm_rows.append({"model_a": a, "model_b": b, **diebold_mariano(preds[a], preds[b], actual)})
    dm = pd.DataFrame(dm_rows, columns=["model_a", "model_b", "stat", "pvalue", "mean_diff", "n"])

    sector = pd.DataFrame({m: rmse_by_sector(p, actual) for m, p in preds.items()}).rename_axis("sektor")

    out = ROOT / "results" / args.freq / "tables"
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.part == "test" else "_val"
    table.to_csv(out / f"karsilastirma{suffix}.csv")
    dm.to_csv(out / f"dm_testleri{suffix}.csv", index=False)
    sector.to_csv(out / f"sektor_rmse{suffix}.csv")

    pd.set_option("display.width", 120)
    print(f"{args.freq} · {args.part} · {len(actual)} dönem · modeller: {', '.join(preds)}\n")
    print(table.round(4).to_string(), "\n")
    if len(dm):
        print("Diebold–Mariano (stat < 0: model_a daha iyi)")
        print(dm.round(4).to_string(index=False))
    print(f"\nYazıldı: {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
