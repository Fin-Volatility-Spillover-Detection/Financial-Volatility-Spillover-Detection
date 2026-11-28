"""AR: eğitim, gecikme seçimi ve tahmin dosyaları (docs/is_bolumu.md, bölüm 2.3).

    python scripts/run_ar.py                  # haftalık
    python scripts/run_ar.py --freq gunluk    # günlük sağlamlık kontrolü

1. Her p ∈ {1, 2, 3, 4} için katsayılar eğitim döneminde tahmin edilir.
2. Her p doğrulama döneminde RMSE ile ölçülür; en düşük RMSE'yi veren p seçilir (bütün sektörler için tek p).
   Bu adımda test dönemi verisi okunmaz bile.
3. Seçilen p'nin (eğitim katsayılarıyla) doğrulama ve test tahminleri yazılır.

Yazar: results/<freq>/predictions/ar_val.csv, ar_test.csv · results/<freq>/params/ar.json
"""

import argparse
import json

from sektorgraf.config import ROOT, load_config
from sektorgraf.data.splits import load_processed, split
from sektorgraf.evaluation.metrics import rmse
from sektorgraf.models.ar import fit_ar, predict_ar

P_GRID = [1, 2, 3, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default="haftalik", choices=["haftalik", "gunluk"])
    args = ap.parse_args()

    cfg = load_config()
    artik = load_processed(cfg, f"artik_ff3_{args.freq}_1985.csv")
    parts = split(artik, cfg)
    train, val, test = parts["train"], parts["val"], parts["test"]
    upto_val = artik.loc[: val.index[-1]]          # ayar seçiminde test dönemi görünmez

    val_rmse, coefs = {}, {}
    for p in P_GRID:
        coefs[p] = fit_ar(train, p)
        val_rmse[p] = rmse(predict_ar(upto_val, coefs[p], val.index), val)
    best = min(val_rmse, key=val_rmse.get)

    out = ROOT / "results" / args.freq
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "params").mkdir(parents=True, exist_ok=True)
    predict_ar(upto_val, coefs[best], val.index).to_csv(out / "predictions" / "ar_val.csv")
    predict_ar(artik, coefs[best], test.index).to_csv(out / "predictions" / "ar_test.csv")

    params = {
        "model": "ar",
        "freq": args.freq,
        "secilen_p": best,
        "val_rmse": val_rmse[best],
        "val_rmse_grid": {str(p): r for p, r in val_rmse.items()},
        "egitim_donem_sayisi": len(train),
    }
    (out / "params" / "ar.json").write_text(json.dumps(params, indent=2, ensure_ascii=False) + "\n")

    print(f"ar · {args.freq} · doğrulama RMSE: " + ", ".join(f"p={p}: {r:.4f}" for p, r in val_rmse.items()))
    print(f"seçilen p = {best} · yazıldı: {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
