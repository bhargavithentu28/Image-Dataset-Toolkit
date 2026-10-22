"""
toolkit/config.py
─────────────────
Configuration loading with sensible defaults.
All callers receive a plain dict so they stay decoupled from this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ── Hard-coded defaults (overridden by config.yaml then by CLI flags) ──────────
_DEFAULTS: dict[str, Any] = {
    "input_dir": "data/raw",
    "output_dir": "data/processed",
    "reports_dir": "reports",
    "supported_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"],
    "preprocessing": {
        "target_width": 640,
        "target_height": 640,
        "mode": "letterbox",
        "pad_color": [114, 114, 114],
        "interpolation_upscale": "INTER_CUBIC",
        "interpolation_downscale": "INTER_AREA",
        "preserve_alpha": True,
    },
    "splitting": {
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "random_seed": 42,
        "stratify": True,
    },
    "duplicates": {
        "enable_perceptual_hash": True,
        "perceptual_hash_threshold": 10,
    },
    "annotations": {
        "format": "yolo",
        "labels_dir": "data/labels",
        "coco_json": "data/annotations.json",
        "class_names_file": "data/classes.txt",
    },
    "logging": {
        "level": "INFO",
        "log_file": "toolkit.log",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load configuration from *config_path* (YAML), falling back to *config.yaml*
    next to this file, then to built-in defaults.

    Parameters
    ----------
    config_path:
        Explicit path to a YAML config file.  Pass *None* to use defaults.

    Returns
    -------
    dict
        Merged configuration mapping.
    """
    cfg = dict(_DEFAULTS)

    candidates: list[Path] = []
    if config_path is not None:
        candidates.append(Path(config_path))
    # Sibling config.yaml
    candidates.append(Path(__file__).parent.parent / "config.yaml")

    for candidate in candidates:
        if candidate.is_file():
            try:
                with candidate.open("r", encoding="utf-8") as fh:
                    loaded = yaml.safe_load(fh) or {}
                cfg = _deep_merge(cfg, loaded)
                logger.debug("Loaded config from %s", candidate)
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not read config %s: %s", candidate, exc)

    return cfg


def setup_logging(cfg: dict[str, Any]) -> None:
    """Configure the root logger from *cfg['logging']*."""
    log_cfg = cfg.get("logging", {})
    level_name: str = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file: str | None = log_cfg.get("log_file")

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
