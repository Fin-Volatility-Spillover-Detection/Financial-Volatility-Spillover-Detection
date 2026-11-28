"""Zero tahmin dosyalarını yazar (docs/is_bolumu.md, bölüm 2.2).

    python scripts/run_zero.py                  # haftalık
    python scripts/run_zero.py --freq gunluk    # günlük sağlamlık kontrolü

Yazar: results/<freq>/predictions/zero_val.csv, zero_test.csv · results/<freq>/params/zero.json
"""

import argparse
import json

from sektorgraf.config import ROOT, load_config
from sektorgraf.data.splits import load_processed, split
from sektorgraf.evaluation.metrics import rmse
from sektorgraf.models.zero import predict_zero


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default="haftalik", choices=["haftalik", "gunluk"])
    args = ap.parse_args()

    cfg = load_config()
    parts = split(load_processed(cfg, f"artik_ff3_{args.freq}_1985.csv"), cfg)

    out = ROOT / "results" / args.freq
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "params").mkdir(parents=True, exist_ok=True)

    for part in ("val", "test"):
        predict_zero(parts[part]).to_csv(out / "predictions" / f"zero_{part}.csv")

    val_rmse = rmse(predict_zero(parts["val"]), parts["val"])
    params = {"model": "zero", "freq": args.freq, "ayar": None, "val_rmse": val_rmse}
    (out / "params" / "zero.json").write_text(json.dumps(params, indent=2, ensure_ascii=False) + "\n")
    print(f"zero · {args.freq} · doğrulama RMSE {val_rmse:.4f} · yazıldı: {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
