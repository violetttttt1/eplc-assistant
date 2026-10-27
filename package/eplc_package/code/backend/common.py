"""Shared helpers for the EPLC backend: package root, config, hashing."""

from __future__ import annotations

import hashlib
import os

import yaml


def package_root() -> str:
    # code/backend/ -> code -> package root
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config() -> dict:
    cfg_path = os.path.join(package_root(), "code", "backend", "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.yaml not found at {cfg_path}")
    return yaml.safe_load(open(cfg_path))


def sha256_files(paths: list[str]) -> str:
    """Deterministic fingerprint of file contents (order-independent input, stable output)."""
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(os.path.basename(p).encode("utf-8"))
        with open(p, "rb") as fh:
            for block in iter(lambda: fh.read(65536), b""):
                h.update(block)
    return h.hexdigest()
