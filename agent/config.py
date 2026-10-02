"""Paths and persisted settings for EzeAi Comfy Agent.

Settings live in user_data/settings.json beside the package so a node update
never overwrites them. Every write is atomic (temp file + replace).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
USER_DATA = Path(os.environ.get("EZEAI_AGENT_DATA") or (PACKAGE_ROOT / "user_data"))
NODE_LLM_DIR = PACKAGE_ROOT / "llm"

DEFAULTS: dict[str, Any] = {
    # provider selection
    "ollama_url": "http://127.0.0.1:11434",
    "llama_server_exe": "",          # blank = Ollama's bundled llama-server.exe
    "external_llama_url": "",        # an already-running llama.cpp / LM Studio server
    "extra_model_dirs": [],
    "allow_remote_hosts": False,     # local-first: refuse non-loopback model servers
    # agent behaviour
    "agent_model": "",               # blank = best available tool-capable model
    "vision_model": "",              # blank = best available vision model
    "permission_mode": "ask",        # inspect | ask | auto
    "max_auto_cycles": 3,            # autonomous build -> run -> review cycles per request
    "share_canvas_view": True,       # send a canvas screenshot so the agent can see it
    "max_rounds": 12,
    "temperature": 0.3,
    "think": False,
    "keep_alive": "2m",
    "context_tokens": 16384,
    "free_vram_before_queue": True,  # unload the LLM before a render is queued
    "offload_between_steps": True,   # unload image models after a run, vision model after use
    # Hard limits. Measured: a 4B model degenerated into repetition and
    # generated 8000+ tokens, holding Ollama's only slot so every later
    # request queued behind it.
    "max_tokens": 1536,
    "run_timeout_s": 240,
}

_LOCK = threading.Lock()


def settings_path() -> Path:
    return USER_DATA / "settings.json"


def load_settings() -> dict[str, Any]:
    path = settings_path()
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
    merged = dict(DEFAULTS)
    merged.update({k: v for k, v in data.items() if k in DEFAULTS})
    return merged


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent),
                                         delete=False, suffix=".tmp")
    try:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        for attempt in range(8):
            try:
                os.replace(handle.name, path)
                return
            except PermissionError:  # Windows: destination briefly held open
                import time

                time.sleep(0.02 * (attempt + 1))
        os.replace(handle.name, path)
    except Exception:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise


def save_settings(update: dict[str, Any]) -> dict[str, Any]:
    """Merge known keys only; unknown keys are ignored, never stored."""
    with _LOCK:
        current = load_settings()
        for key, value in (update or {}).items():
            if key not in DEFAULTS:
                continue
            expected = type(DEFAULTS[key])
            if expected is bool:
                value = bool(value)
            elif expected is int:
                value = int(value)
            elif expected is float:
                value = float(value)
            elif expected is list:
                value = [str(v) for v in (value or []) if str(v).strip()]
            else:
                value = str(value)
            current[key] = value
        if current["permission_mode"] not in ("inspect", "ask", "auto"):
            current["permission_mode"] = "ask"
        current["max_auto_cycles"] = max(1, min(int(current["max_auto_cycles"]), 8))
        current["max_rounds"] = max(1, min(int(current["max_rounds"]), 20))
        atomic_write_json(settings_path(), current)
        return current
