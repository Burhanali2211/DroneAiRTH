"""config_loader.py — Centralized configuration management."""
import yaml
from pathlib import Path

_cfg = None

def get_config(path: str = None) -> dict:
    """Load and cache config.yaml from project root."""
    global _cfg
    if _cfg is None:
        p = Path(path) if path else Path(__file__).parent.parent / 'config.yaml'
        _cfg = yaml.safe_load(p.read_text())
    return _cfg
