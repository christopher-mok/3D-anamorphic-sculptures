"""YAML configuration with deep-merge presets and dotted overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

PROJECT_ROOT = Path(os.environ.get("SCULPTURE_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR = PROJECT_ROOT / "configs"
PRESETS = ("fast", "default", "high_quality")


class Config(dict):
    """A dict with attribute access for nested keys (``cfg.beam.beam_width``)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - mirrors attribute semantics
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __deepcopy__(self, memo):
        return Config({k: copy.deepcopy(v, memo) for k, v in self.items()})

    def to_dict(self) -> dict:
        return _plain(self)


def _wrap(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _plain(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def deep_merge(base: Mapping, override: Mapping) -> dict:
    out = {k: copy.deepcopy(v) for k, v in base.items()}
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_scalar(text: str) -> Any:
    return yaml.safe_load(text)


def apply_dotted(cfg: dict, assignments: Iterable[str]) -> dict:
    """Apply ``a.b.c=value`` overrides (values parsed as YAML)."""
    cfg = copy.deepcopy(cfg)
    for item in assignments:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_scalar(value)
    return cfg


def resolve_config_path(name_or_path: str | os.PathLike | None) -> Path:
    if name_or_path is None:
        return CONFIG_DIR / "default.yaml"
    p = Path(name_or_path)
    if p.exists():
        return p
    candidate = CONFIG_DIR / f"{name_or_path}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"config {name_or_path!r} not found")


def load_config(
    name_or_path: str | os.PathLike | None = None,
    overrides: Mapping | None = None,
    dotted: Iterable[str] = (),
) -> Config:
    """Load default.yaml, merge a preset/config file, nested overrides and dotted overrides."""
    with open(CONFIG_DIR / "default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    path = resolve_config_path(name_or_path)
    if path.resolve() != (CONFIG_DIR / "default.yaml").resolve():
        with open(path, "r", encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    cfg = apply_dotted(cfg, dotted)
    return _wrap(cfg)
