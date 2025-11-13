import yaml
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
print(BASE_DIR)
CONFIG_PATH = BASE_DIR / "config.yaml"
print(f"！Loading config from: {CONFIG_PATH}")

class Config:
    def __init__(self, path=CONFIG_PATH):
        with open(path) as f:
            data = yaml.safe_load(f)   # safe_load 会把 3e-4 当作 float
        self._cfg = data

    def __getattr__(self, name):
        # 支持 cfg.env, cfg.train, cfg.data, cfg.export
        if name in self._cfg:
            return self._cfg[name]
        raise AttributeError(f"No such config field: {name}")

    def __getitem__(self, key):
        return self._cfg[key]
