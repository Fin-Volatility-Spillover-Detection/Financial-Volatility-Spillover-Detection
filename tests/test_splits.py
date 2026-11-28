"""Ortak ayrımın garantileri: çakışma yok, boşluk yok, sıra doğru."""

from sektorgraf.config import load_config
from sektorgraf.data.splits import load_processed, split


def test_split_is_ordered_disjoint_and_complete():
    cfg = load_config()
    for f in ["artik_ff3_haftalik_1985.csv", "artik_ff3_gunluk_1985.csv", "artik_ff3_aylik_1985.csv"]:
        df = load_processed(cfg, f)
        p = split(df, cfg)
        assert p["train"].index.max() < p["val"].index.min()
        assert p["val"].index.max() < p["test"].index.min()
        assert sum(len(x) for x in p.values()) == len(df)        # hiçbir satır dışarıda kalmıyor


def test_last_partial_period_dropped_except_daily():
    cfg = load_config()
    assert str(load_processed(cfg, "artik_ff3_haftalik_1985.csv").index[-1].date()) == "2026-09-25"
    assert str(load_processed(cfg, "artik_ff3_gunluk_1985.csv").index[-1].date()) == "2026-09-30"
