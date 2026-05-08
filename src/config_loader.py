"""config_loader.py — Centralized configuration management."""
import threading
import yaml
from pathlib import Path

_cfg  = None
_lock = threading.Lock()

def get_config(path: str = None) -> dict:
    """Load and cache config.yaml. Thread-safe; raises RuntimeError on missing file."""
    global _cfg
    if _cfg is not None:
        return _cfg
    with _lock:
        if _cfg is not None:   # re-check after acquiring lock
            return _cfg
        p = Path(path) if path else Path(__file__).parent.parent / 'config.yaml'
        if not p.exists():
            raise RuntimeError(
                f"config.yaml not found at {p.resolve()}. "
                "Copy config.yaml to the project root before running."
            )
        _cfg = yaml.safe_load(p.read_text())
    return _cfg
