from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/config.yaml") -> dict:
    """configs/config.yaml dosyasını okur. Göreli yollar repo köküne göre çözülür."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, val in cfg["paths"].items():
        cfg["paths"][key] = ROOT / val
    return cfg
