"""Shared helpers: config/params loading, logging, HTML cleaning, IO."""
from __future__ import annotations
import logging, sys, re, html
from pathlib import Path
import yaml


def get_logger(name: str = "book_atlas") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_params(path: str = "params.yaml") -> dict:
    return _load_yaml(path)


def load_config(path: str = "config/datasets.yaml") -> dict:
    return _load_yaml(path)


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean_html(text: str | None) -> str:
    if not text:
        return ""
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    return _WS.sub(" ", text).strip()


def ensure_dir(p: str) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def l2_normalize(v):
    import numpy as np
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return v / n
