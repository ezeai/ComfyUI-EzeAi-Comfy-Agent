"""Run GGUF models through llama.cpp's llama-server, one at a time.

Nothing is installed. By default the llama-server.exe that ships inside Ollama
is used (the same recipe Content Studio and Alusi live-verified: --jinja chat
template, CUDA backend from Ollama's cuda_v13/v12 folder). A different
llama-server can be chosen in Settings.

Only one model is resident. `stop()` frees the GPU immediately and is called
before a ComfyUI render is queued, because the LLM and the diffusion model
compete for the same VRAM.
"""

from __future__ import annotations

import atexit
import os
import secrets
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from .config import USER_DATA, load_settings

_RUN: dict[str, Any] = {}
_LOCK = threading.RLock()
_IDLE: threading.Timer | None = None
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _kill_with_parent(proc: subprocess.Popen) -> None:
    """Windows job object: llama-server dies with ComfyUI even if ComfyUI is killed,
    so a stopped session cannot leave a model holding gigabytes of RAM and VRAM."""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if "job" not in _RUN_JOB:
        job = kernel32.CreateJobObjectW(None, None)

        class Limits(ctypes.Structure):
            _fields_ = [("a", ctypes.c_int64), ("b", ctypes.c_int64), ("LimitFlags", wintypes.DWORD),
                        ("c", ctypes.c_size_t), ("d", ctypes.c_size_t), ("e", wintypes.DWORD),
                        ("f", ctypes.c_size_t), ("g", wintypes.DWORD), ("h", wintypes.DWORD)]

        class Extended(ctypes.Structure):
            _fields_ = [("Basic", Limits), ("io", ctypes.c_uint64 * 6), ("p", ctypes.c_size_t * 4)]

        info = Extended()
        info.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
        _RUN_JOB["job"] = job
    kernel32.AssignProcessToJobObject(_RUN_JOB["job"], wintypes.HANDLE(int(proc._handle)))


_RUN_JOB: dict[str, Any] = {}


def llama_server_exe() -> Path | None:
    custom = (load_settings().get("llama_server_exe") or "").strip()
    if custom:
        path = Path(custom)
        return path if path.is_file() else None
    bundled = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "lib" / "ollama" / "llama-server.exe"
    if bundled.is_file():
        return bundled
    for name in ("llama-server.exe", "llama-server"):
        for folder in os.environ.get("PATH", "").split(os.pathsep):
            candidate = Path(folder) / name
            if candidate.is_file():
                return candidate
    return None


def _backend_dir(exe: Path) -> Path | None:
    for name in ("cuda_v13", "cuda_v12"):
        candidate = exe.parent / name
        if (candidate / "ggml-cuda.dll").is_file():
            return candidate
    return None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def status() -> dict[str, Any]:
    with _LOCK:
        proc = _RUN.get("proc")
        alive = bool(proc and proc.poll() is None)
        return {"resident": _RUN.get("ref") if alive else None,
                "port": _RUN.get("port") if alive else None,
                "since": _RUN.get("since") if alive else None}


def stop(reason: str = "") -> bool:
    """Free the GPU now. Returns True when a model was actually stopped."""
    global _IDLE
    with _LOCK:
        if _IDLE:
            _IDLE.cancel()
            _IDLE = None
        proc = _RUN.pop("proc", None)
        _RUN.clear()
    if proc and proc.poll() is None:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True, creationflags=NO_WINDOW)
        return True
    return False


def schedule_stop(seconds: float) -> None:
    global _IDLE
    with _LOCK:
        if _IDLE:
            _IDLE.cancel()
        if seconds <= 3600:
            _IDLE = threading.Timer(max(0.0, seconds), stop)
            _IDLE.daemon = True
            _IDLE.start()


def keep_alive_seconds(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "2m").strip()
    for suffix, mult in (("ms", 0.001), ("s", 1), ("m", 60), ("h", 3600)):
        if text.endswith(suffix) and text[:-len(suffix)].replace(".", "").isdigit():
            return float(text[:-len(suffix)]) * mult
    return float(text) if text.replace(".", "").isdigit() else 120.0


def ensure(model: dict[str, Any], *, context: int = 16384) -> dict[str, Any]:
    """Make this GGUF the resident model and return {port, api_key}."""
    with _LOCK:
        proc = _RUN.get("proc")
        if _RUN.get("ref") == model["ref"] and proc and proc.poll() is None:
            if _IDLE:
                _IDLE.cancel()
            return dict(_RUN)
        stop()
        exe = llama_server_exe()
        if exe is None:
            raise RuntimeError("llama-server was not found. Install Ollama (it bundles one) or set "
                               "its path in EzeAi Comfy Agent > Settings.")
        path = Path(model["path"])
        if not path.is_file():
            raise RuntimeError(f"model file is missing: {path}")
        backend = _backend_dir(exe)
        port, api_key = _free_port(), secrets.token_hex(16)
        log = USER_DATA / "logs" / f"llama-{path.stem[:60]}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        args = [str(exe), "--model", str(path), "--jinja", "--host", "127.0.0.1",
                "--port", str(port), "--no-webui", "-c", str(int(context)), "-np", "1",
                "-ngl", "99", "--flash-attn", "on", "--log-file", str(log)]
        if model.get("mmproj"):
            args += ["--mmproj", str(model["mmproj"])]
        env = {**os.environ, "LLAMA_API_KEY": api_key,
               "PATH": os.pathsep.join(str(x) for x in (backend, exe.parent, os.environ.get("PATH", "")) if x)}
        proc = subprocess.Popen(args, cwd=str(backend or exe.parent), env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                creationflags=NO_WINDOW)
        _kill_with_parent(proc)
        _RUN.update(ref=model["ref"], proc=proc, port=port, api_key=api_key, since=time.time())

    deadline = time.time() + 180
    while True:
        if proc.poll() is not None:
            tail = log.read_text(encoding="utf-8", errors="replace")[-600:] if log.is_file() else ""
            stop()
            last = tail.strip().splitlines()[-1:] if tail.strip() else ["no log output"]
            raise RuntimeError(f"llama-server exited while loading {path.name}: {last[0]}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return dict(_RUN)
        except Exception:  # noqa: BLE001
            pass
        if time.time() > deadline:
            stop()
            raise RuntimeError(f"{path.name} did not load within 3 minutes (see {log})")
        time.sleep(0.4)


atexit.register(stop)
