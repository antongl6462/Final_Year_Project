import yaml
from types import SimpleNamespace

def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg)

def save_config(cfg, path):
    with open(path, 'w') as f:
        yaml.safe_dump(vars(cfg), f)
