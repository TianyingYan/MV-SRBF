"""Experiment tracking with a pluggable backend and a local JSONL fallback.

Supported ``logging.tracker`` backends:

- ``"trackio"`` (default): logs to a local `trackio` store and, when
  ``logging.dashboard.enabled`` is set, launches a local web dashboard on the
  configured host/port for live monitoring during training.
- ``"none"``: only the local ``output_dir/logs/metrics.jsonl`` file is written.

Rank 0 always writes the local JSONL log so runs stay auditable even when
Trackio is disabled or unavailable.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

_cfg: Dict[str, Any] = {}
_run = None
_backend = "none"
_local_log_path: Optional[str] = None
_backend_live = False


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _global_rank() -> int:
    """Best-effort process rank; only rank 0 owns shared tracking outputs."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    for var in ("RANK", "LOCAL_RANK"):
        value = os.environ.get(var)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                return 0
    return 0


def _init_local_logger(cfg: Dict[str, Any]) -> None:
    global _local_log_path
    log_cfg = cfg.get("logging", {})
    output_dir = cfg.get("output_dir", "logs/run")
    log_dir = os.path.join(output_dir, "logs")
    path = os.path.abspath(log_cfg.get("local_log_path", os.path.join(log_dir, "metrics.jsonl")))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "start", "time": time.time()}, default=_json_default) + "\n")
    _local_log_path = path


def _resolve_backend(log_cfg: Dict[str, Any]) -> str:
    """Resolve the configured tracker backend."""
    return str(log_cfg.get("tracker", "none")).strip().lower()


def _resolve_project(log_cfg: Dict[str, Any]) -> str:
    return str(log_cfg.get("project", "MV-SRBF"))


def _resolve_run_name(log_cfg: Dict[str, Any]) -> str:
    """Build a fresh, timestamped run name for this startup.

    Every launch — a fresh run or a breakpoint resume whose training config may have
    changed — must be a NEW trackio run/visualization, never a continuation of an
    earlier one. The timestamp suffix makes each startup uniquely identifiable, and
    ``init_tracker`` also passes ``resume="never"`` so the backend never merges runs.
    """
    base = log_cfg.get("run_name")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{base}-{stamp}" if base else f"run-{stamp}"


def _launch_trackio_dashboard(log_cfg: Dict[str, Any], project: str) -> None:
    """Start trackio's local web UI without blocking the training thread."""
    dash = log_cfg.get("dashboard", {}) or {}
    if not dash.get("enabled", True):
        return
    try:
        import trackio

        url = trackio.show(
            project=project,
            host=str(dash.get("host", "127.0.0.1")),
            server_port=int(dash["port"]) if dash.get("port") is not None else None,
            open_browser=bool(dash.get("open_browser", False)),
            share=bool(dash.get("share", False)),
            block_thread=False,
        )
        _write_local(
            {
                "event": "dashboard_started",
                "host": dash.get("host", "127.0.0.1"),
                "port": dash.get("port"),
                "url": str(url) if url is not None else None,
            },
            step=0,
        )
    except Exception as exc:  # never let monitoring crash training
        _write_local({"event": "dashboard_unavailable", "reason": str(exc)}, step=0)


def init_tracker(cfg: Dict[str, Any], merged_flat: Optional[Dict[str, Any]] = None) -> bool:
    """Initialize the tracking backend. Returns True iff a remote backend is live."""
    global _run, _cfg, _backend, _backend_live, _local_log_path
    _cfg = cfg
    _run = None
    _backend = "none"
    _backend_live = False
    _local_log_path = None

    # Rank 0 owns the shared JSONL file, tracker run, and dashboard port.
    if _global_rank() != 0:
        return False

    _init_local_logger(cfg)

    log_cfg = cfg.get("logging", {})
    backend = _resolve_backend(log_cfg)
    if backend in ("none", ""):
        return False

    project = _resolve_project(log_cfg)
    run_name = _resolve_run_name(log_cfg)

    # Strip keys starting with '_' (e.g. '_config_files') — tracking backends
    # reserve the '_' prefix for internal use and will reject such keys.
    raw_config = merged_flat or cfg
    clean_config = {k: v for k, v in raw_config.items() if not k.startswith("_")}

    if backend == "trackio":
        try:
            import trackio

            # resume="never" -> always a new run/visualization for this startup.
            _run = trackio.init(project=project, name=run_name, config=clean_config, resume="never")
            _backend = "trackio"
            _backend_live = True
            _launch_trackio_dashboard(log_cfg, project)
            return True
        except Exception as exc:
            _write_local({"event": "trackio_unavailable", "reason": str(exc)}, step=0)
            return False

    _write_local({"event": "tracker_unknown", "reason": f"unknown tracker '{backend}'"}, step=0)
    return False


def _write_local(data: Dict[str, Any], step: int) -> None:
    if not _local_log_path:
        return
    record = {"step": int(step), "time": time.time(), **data}
    with open(_local_log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default) + "\n")


def log_scalars(data: Dict[str, Any], step: int) -> None:
    # The local JSONL file is the authoritative run log. Remote trackers are
    # optional mirrors and must never suppress local persistence.
    _write_local(data, step)
    if not _backend_live:
        return
    try:
        if _backend == "trackio":
            import trackio

            trackio.log(data, step=step)
    except Exception as exc:
        _write_local({"event": "tracker_log_failed", "reason": str(exc)}, step)


def log_event(event: str, step: int = 0, **fields: Any) -> None:
    """Append a structured event line to the local ``metrics.jsonl`` log.

    Unlike :func:`log_scalars`, this always writes a breadcrumb to the local JSONL
    file so non-scalar context (e.g. a breakpoint checkpoint path) is preserved even
    when Trackio is active. When Trackio is live, scalar fields are also
    mirrored as ``{event}/{key}`` metrics.
    """
    _write_local({"event": event, **fields}, step)
    if not _backend_live:
        return
    scalar_fields = {
        f"{event}/{key}": value
        for key, value in fields.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if not scalar_fields:
        return
    try:
        if _backend == "trackio":
            import trackio

            trackio.log(scalar_fields, step=step)
    except Exception as exc:
        _write_local({"event": "tracker_log_failed", "reason": str(exc)}, step)


def finish() -> None:
    _write_local({"event": "finish"}, step=0)
    if not _backend_live:
        return
    try:
        if _backend == "trackio":
            import trackio

            trackio.finish()
    except Exception as exc:
        _write_local({"event": "tracker_finish_failed", "reason": str(exc)}, step=0)
