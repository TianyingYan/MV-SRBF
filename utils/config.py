import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_runtime_path(path_value: str, path_base: str | Path | None = None) -> str:
    """Resolve a local config path from the directory where the config was loaded."""
    raw = str(path_value or "").strip()
    if not raw:
        return ""
    path = Path(os.path.expandvars(os.path.expanduser(raw)))
    if path.is_absolute():
        return str(path.resolve())
    base = Path(path_base).expanduser() if path_base else Path.cwd()
    return str((base / path).resolve())


def deep_merge(a: Dict, b: Dict) -> Dict:
    """Recursively merge dict b into dict a; b overrides a."""
    for k, v in b.items():
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            deep_merge(a[k], v)
        else:
            a[k] = v
    return a


def _parse_override_value(raw: str) -> Any:
    """Parse a command-line override value using YAML scalar/list/dict rules."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid --parameter value {raw!r}: {exc}") from exc


def _set_by_dotted_path(cfg: Dict[str, Any], path: str, value: Any) -> None:
    """Set cfg[a][b][c] from a dotted path, creating dictionaries as needed."""
    keys = [part.strip() for part in str(path).split(".") if part.strip()]
    if not keys:
        raise ValueError("--parameter path cannot be empty.")
    cur: Dict[str, Any] = cfg
    for key in keys[:-1]:
        next_value = cur.get(key)
        if next_value is None:
            next_value = {}
            cur[key] = next_value
        if not isinstance(next_value, dict):
            raise ValueError(
                f"Cannot apply --parameter {path!r}: {key!r} already exists and is not a mapping."
            )
        cur = next_value
    cur[keys[-1]] = value


def apply_parameter_overrides(cfg: Dict[str, Any], parameter_specs: Iterable[str] | None = None) -> Dict[str, Any]:
    """
    Apply command-line YAML overrides.

    Supported forms:
      * ``a.b.c=value`` sets one dotted path. Missing intermediate mappings are created.
      * ``{a: {b: 1}}`` merges a YAML/JSON mapping into the config.
    """
    specs = list(parameter_specs or [])
    if not specs:
        return cfg
    for spec in specs:
        text = str(spec).strip()
        if not text:
            continue
        if "=" in text and not text.lstrip().startswith(("{", "[")):
            path, raw_value = text.split("=", 1)
            _set_by_dotted_path(cfg, path, _parse_override_value(raw_value))
        else:
            value = _parse_override_value(text)
            if not isinstance(value, dict):
                raise ValueError(
                    "--parameter without KEY=VALUE must be a YAML mapping, "
                    f"got {type(value).__name__}."
                )
            deep_merge(cfg, value)
    cfg.setdefault("_parameter_overrides", []).extend(specs)
    return cfg


def _resolve_config_path(config_name: str, config_dir: str = "configs") -> Path:
    p = Path(config_name)
    if p.is_file() and p.suffix.lower() in (".yaml", ".yml"):
        config_path = p.resolve()
    else:
        raw_base = Path(config_dir)
        base_dirs = [raw_base] if raw_base.is_absolute() else [PROJECT_ROOT / raw_base, Path(sys.prefix) / raw_base]
        candidates = []
        for base_dir in base_dirs:
            if str(config_name).endswith((".yaml", ".yml")):
                candidates.append((base_dir / config_name).resolve())
            else:
                candidates.extend(
                    [
                        (base_dir / f"{config_name}.yaml").resolve(),
                        (base_dir / "experiments" / f"{config_name}.yaml").resolve(),
                        (base_dir / "data" / f"{config_name}.yaml").resolve(),
                    ]
                )
        config_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return config_path


def _load_config(config_path: Path) -> Tuple[Dict[str, Any], List[str]]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    source_files: List[str] = [str(config_path)]
    if isinstance(cfg, dict) and "defaults" in cfg:
        base_cfg = {}
        current_dir = config_path.parent

        for default_spec in cfg["defaults"]:
            rel_path = Path(default_spec).with_suffix(".yaml")
            abs_path = (current_dir / rel_path).resolve()

            if not abs_path.exists():
                raise FileNotFoundError(
                    f"Default config '{default_spec}' not found. "
                    f"Searched at: {abs_path}"
                )

            sub_cfg, sub_sources = _load_config(abs_path)
            base_cfg = deep_merge(base_cfg, sub_cfg)
            source_files.extend(sub_sources)

        del cfg["defaults"]
        cfg = deep_merge(base_cfg, cfg)

    return cfg, source_files


def get_config(config_name: str, config_dir: str = "configs", include_metadata: bool = False) -> Dict[str, Any]:
    """
    Load a YAML config file and recursively merge entries listed in defaults.

    Args:
        config_name: Absolute/relative YAML path, or a config name under config_dir.
        config_dir: Directory used when config_name is not a direct YAML path.
        include_metadata: Attach source YAML paths under `_config_files`.

    Returns:
        Merged config dictionary. Relative runtime asset paths are resolved from
        the captured `_path_base`, which is the current working directory at
        config-load time.
    """
    path_base = Path.cwd().resolve()
    config_path = _resolve_config_path(config_name, config_dir)
    cfg, source_files = _load_config(config_path)
    cfg["_path_base"] = str(path_base)
    if include_metadata:
        data_parts = {"configs", "data"}
        cfg["_config_files"] = {
            "entry": str(config_path),
            "all": source_files,
            "data": [p for p in source_files if data_parts.issubset(set(Path(p).parts))],
        }
    return cfg


def archive_run_config(cfg: Dict[str, Any], phase: str) -> None:
    """Archive the effective run YAML config under the log directory."""
    output_dir = Path(cfg.get("output_dir", "logs/run"))
    archive_dir = output_dir / "configs" / phase
    archive_dir.mkdir(parents=True, exist_ok=True)

    clean_cfg = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    with open(archive_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(clean_cfg, handle, sort_keys=False, allow_unicode=True)
