"""Load project configuration files without external dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

from .schema import PathsConfig


DEFAULT_PATHS_FILE = Path("configs/paths.yaml")


def load_paths_config(path: Path | str | None = None) -> PathsConfig:
    """Load the path convention used by SkillRQ.

    The M0 configuration intentionally supports a small flat YAML subset:
    `key: value` pairs with optional comments. This keeps project startup free
    of third-party dependencies while remaining easy to replace with PyYAML or
    OmegaConf later.
    """

    config_path = Path(path) if path is not None else DEFAULT_PATHS_FILE
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    mapping = dict(PathsConfig.defaults())
    if config_path.exists():
        mapping.update(_parse_flat_yaml(config_path))

    return PathsConfig.from_mapping(mapping, project_root=Path.cwd())


def _parse_flat_yaml(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Invalid config line {line_number} in {path}: {raw_line!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = _unquote(value.strip())
        if not key:
            raise ValueError(f"Empty config key on line {line_number} in {path}")
        values[key] = value
    return values


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value

