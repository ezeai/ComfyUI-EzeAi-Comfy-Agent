"""Discover the local language models this PC can actually run.

Three sources, one list:

  ollama:<name>         models served by a local Ollama
  gguf:<absolute path>  GGUF files run through llama.cpp's llama-server
  server:<model id>     an already-running OpenAI-compatible local server

Capabilities are MEASURED, not guessed from a filename: Ollama reports them in
/api/show, and for a GGUF the chat template embedded in the file says whether
it understands tools. A GGUF counts as vision-capable only when a matching
mmproj projector sits beside it.
"""

from __future__ import annotations

import json
import os
import re
import struct
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import COMFY_ROOT, NODE_LLM_DIR, load_settings

_CACHE: dict[str, Any] = {}
_LOCK = threading.Lock()


# --- minimal GGUF metadata reader ------------------------------------------
# GGUF stores key/value metadata before the tensors. Only a handful of keys are
# needed, so this reads the header and stops; it never loads weights.

_GGUF_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
                 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_WANTED = ("general.architecture", "general.name", "tokenizer.chat_template")


def _read_str(f) -> str:
    (length,) = struct.unpack("<Q", f.read(8))
    if length > 64 * 1024 * 1024:
        raise ValueError("implausible GGUF string length")
    return f.read(length).decode("utf-8", errors="replace")


def _skip_value(f, vtype: int) -> None:
    if vtype in _GGUF_SCALARS:
        f.seek(struct.calcsize(_GGUF_SCALARS[vtype]), 1)
    elif vtype == 8:
        (length,) = struct.unpack("<Q", f.read(8))
        f.seek(length, 1)
    elif vtype == 9:
        (itype,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        if itype in _GGUF_SCALARS:
            f.seek(count * struct.calcsize(_GGUF_SCALARS[itype]), 1)
        else:
            for _ in range(count):
                _skip_value(f, itype)
    else:
        raise ValueError(f"unknown GGUF value type {vtype}")


def read_gguf_metadata(path: Path) -> dict[str, Any]:
    """Return architecture, name, context length and chat template, cached."""
    stat = path.stat()
    key = f"gguf:{path}:{stat.st_mtime_ns}:{stat.st_size}"
    if key in _CACHE:
        return _CACHE[key]
    meta: dict[str, Any] = {}
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        (version,) = struct.unpack("<I", f.read(4))
        if version < 2:
            raise ValueError(f"GGUF version {version} not supported")
        f.read(8)  # tensor count
        (kv_count,) = struct.unpack("<Q", f.read(8))
        for _ in range(kv_count):
            name = _read_str(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            if (name in _WANTED or name.endswith(".context_length")) and vtype in (4, 5, 8, 10, 11):
                if vtype == 8:
                    meta[name] = _read_str(f)
                else:
                    meta[name] = struct.unpack(_GGUF_SCALARS[vtype], f.read(
                        struct.calcsize(_GGUF_SCALARS[vtype])))[0]
            else:
                _skip_value(f, vtype)
            if all(k in meta for k in _WANTED) and any(
                    k.endswith(".context_length") for k in meta):
                break
    arch = str(meta.get("general.architecture", ""))
    ctx = next((v for k, v in meta.items() if k.endswith(".context_length")), None)
    template = str(meta.get("tokenizer.chat_template", ""))
    info = {
        "architecture": arch,
        "name": meta.get("general.name") or path.stem,
        "context_length": ctx,
        "has_template": bool(template),
        # A template that renders a tools block is the honest signal that the
        # model was trained to call tools.
        "tools": bool(re.search(r"\btools?\b", template)),
        "thinking": "<think>" in template or "enable_thinking" in template,
    }
    _CACHE[key] = info
    return info


# --- GGUF discovery ---------------------------------------------------------


def model_dirs() -> list[Path]:
    """Every folder that is scanned for GGUF files, in priority order."""
    settings = load_settings()
    candidates = [NODE_LLM_DIR, COMFY_ROOT / "models" / "LLM", COMFY_ROOT / "models" / "llm_gguf"]
    try:  # honour extra_model_paths.yaml registrations if ComfyUI exposes them
        import folder_paths  # type: ignore

        for folder in ("LLM", "llm", "llm_gguf"):
            if folder in folder_paths.folder_names_and_paths:
                candidates.extend(Path(p) for p in folder_paths.folder_names_and_paths[folder][0])
    except Exception:
        pass
    candidates.extend(Path(p) for p in settings.get("extra_model_dirs") or [])
    seen, out = set(), []
    for directory in candidates:
        try:
            resolved = directory.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _is_projector(path: Path) -> bool:
    """A vision projector, not a language model.

    The file name is only a hint ("Qwen3.8-27B-mmproj-F16.gguf" does not start
    with "mmproj"); the metadata is authoritative: projectors declare the
    `clip` architecture.
    """
    if "mmproj" in path.name.lower():
        return True
    try:
        return read_gguf_metadata(path).get("architecture") == "clip"
    except Exception:  # noqa: BLE001
        return False


def _pair_projector(model: Path, projectors: list[Path]) -> Path | None:
    """Choose the mmproj that belongs to this model, or None.

    Same folder only; when several are present, the one sharing the longest
    name stem with the model wins, so a Gemma model never borrows a Qwen
    projector.
    """
    local = [p for p in projectors if p.parent == model.parent]
    if not local:
        return None
    if len(local) == 1 and len(list(model.parent.glob("*.gguf"))) <= 2:
        return local[0]

    def score(p: Path) -> int:
        a = re.sub(r"[^a-z0-9]", "", model.stem.lower())
        b = re.sub(r"[^a-z0-9]", "", p.stem.lower().replace("mmproj", ""))
        best = 0
        for size in range(min(len(a), len(b)), 3, -1):
            if any(b[i:i + size] in a for i in range(len(b) - size + 1)):
                best = size
                break
        return best

    best = max(local, key=score)
    return best if score(best) >= 6 else None


def discover_gguf() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for directory in model_dirs():
        files = [p for p in directory.rglob("*.gguf") if p.is_file()]
        projectors = [p for p in files if _is_projector(p)]
        for path in sorted(files):
            if path in projectors:
                continue
            try:
                meta = read_gguf_metadata(path)
            except Exception as exc:  # noqa: BLE001
                meta = {"architecture": "", "name": path.stem, "context_length": None,
                        "tools": False, "thinking": False, "error": str(exc)}
            # the same model file mirrored in two folders is listed once
            dedupe = f"{path.name}:{path.stat().st_size}"
            if dedupe in seen_names:
                continue
            seen_names.add(dedupe)
            projector = _pair_projector(path, projectors)
            models.append({
                "ref": f"gguf:{path}",
                "provider": "llama.cpp",
                "name": path.stem,
                "label": meta.get("name") or path.stem,
                "family": meta.get("architecture", ""),
                "size_gb": round(path.stat().st_size / 1e9, 1),
                "context": meta.get("context_length"),
                "path": str(path),
                "mmproj": str(projector) if projector else None,
                "caps": {
                    "tools": bool(meta.get("tools")),
                    "vision": projector is not None,
                    "thinking": bool(meta.get("thinking")),
                    "audio": False,
                },
                "source": str(directory),
                "error": meta.get("error"),
            })
    return models


# --- Ollama discovery -------------------------------------------------------


def _http_json(url: str, body: dict | None = None, timeout: float = 8) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def discover_ollama() -> tuple[list[dict[str, Any]], str | None]:
    base = load_settings()["ollama_url"].rstrip("/")
    try:
        tags = _http_json(f"{base}/api/tags", timeout=12)
    except Exception as exc:  # noqa: BLE001
        return [], f"Ollama not reachable at {base}: {exc}"
    models = []
    for entry in tags.get("models", []):
        name = entry.get("name", "")
        digest = entry.get("digest", name)
        key = f"ollama:{digest}"
        show = _CACHE.get(key)
        if show is None:
            try:
                show = _http_json(f"{base}/api/show", {"model": name}, timeout=10)
            except Exception:  # noqa: BLE001
                show = {}
            _CACHE[key] = show
        caps = set(show.get("capabilities") or [])
        info = show.get("model_info") or {}
        ctx = next((v for k, v in info.items() if k.endswith(".context_length")), None)
        details = entry.get("details") or {}
        if "embedding" in caps and "completion" not in caps:
            continue  # embedding-only models cannot chat
        models.append({
            "ref": f"ollama:{name}",
            "provider": "ollama",
            "name": name,
            "label": name,
            "family": details.get("family", ""),
            "params": details.get("parameter_size", ""),
            "size_gb": round(entry.get("size", 0) / 1e9, 1),
            "context": ctx,
            "caps": {"tools": "tools" in caps, "vision": "vision" in caps,
                     "thinking": "thinking" in caps, "audio": "audio" in caps},
            "source": base,
        })
    return models, None


def discover_external() -> tuple[list[dict[str, Any]], str | None]:
    url = (load_settings().get("external_llama_url") or "").rstrip("/")
    if not url:
        return [], None
    try:
        data = _http_json(f"{url}/v1/models", timeout=4)
    except Exception as exc:  # noqa: BLE001
        return [], f"external server not reachable at {url}: {exc}"
    out = []
    for entry in data.get("data", []):
        mid = entry.get("id", "model")
        out.append({"ref": f"server:{mid}", "provider": "openai-compatible", "name": mid,
                    "label": f"{mid} (server)", "family": "", "size_gb": None, "context": None,
                    # an external server's tool support is only known by probing it
                    "caps": {"tools": True, "vision": False, "thinking": False, "audio": False},
                    "source": url})
    return out, None


# --- the combined catalogue -------------------------------------------------


def catalogue(refresh: bool = False) -> dict[str, Any]:
    with _LOCK:
        if not refresh and "catalogue" in _CACHE and time.time() - _CACHE["catalogue_at"] < 30:
            return _CACHE["catalogue"]
        ollama, ollama_err = discover_ollama()
        external, external_err = discover_external()
        gguf = discover_gguf()
        from .runtime import llama_server_exe

        exe = llama_server_exe()
        for m in gguf:
            m["runnable"] = exe is not None
        result = {
            "models": ollama + gguf + external,
            "ollama_error": ollama_err,
            "external_error": external_err,
            "llama_server": str(exe) if exe else None,
            "dirs": [str(d) for d in model_dirs()],
            "recommended": recommend(ollama + gguf + external),
        }
        _CACHE["catalogue"] = result
        _CACHE["catalogue_at"] = time.time()
        return result


# Ordered preferences, matched against the model name. Spark X2.5 was trained
# for tool use and agent loops and is the fastest tool-capable model here, so
# it leads for the agent; Qwen 3.5 is preferred for vision.
_AGENT_PREFERENCE = ("spark", "qwen3.5:9b", "cs-operator:", "qwen3.5", "qwen3", "gemma4", "llama3.1")
_VISION_PREFERENCE = ("qwen3.5:9b", "cs-operator:", "qwen3.5", "gemma4", "gemma-4")


def recommend(models: list[dict[str, Any]]) -> dict[str, str | None]:
    def pick(prefs, need):
        usable = [m for m in models if all(m["caps"].get(c) for c in need)]
        for pref in prefs:
            for m in usable:
                if pref in m["name"].lower() and (m["provider"] == "ollama"):
                    return m["ref"]
            for m in usable:
                if pref in m["name"].lower():
                    return m["ref"]
        return usable[0]["ref"] if usable else None

    return {"agent": pick(_AGENT_PREFERENCE, ("tools",)),
            "vision": pick(_VISION_PREFERENCE, ("vision",))}


def resolve(ref: str | None, *, need: tuple[str, ...] = ()) -> dict[str, Any]:
    """Turn a model reference (or blank = recommended) into a catalogue entry.

    A cached catalogue can be stale (Ollama was busy loading a model when it
    was taken), so a miss triggers one fresh discovery before failing.
    """
    try:
        return _resolve(catalogue(), ref, need)
    except LookupError:
        cat = catalogue(refresh=True)
        try:
            return _resolve(cat, ref, need)
        except LookupError as exc:
            reasons = [r for r in (cat.get("ollama_error"), cat.get("external_error")) if r]
            raise ValueError(str(exc) + (f" ({'; '.join(reasons)})" if reasons else "")) from None


def _resolve(cat: dict[str, Any], ref: str | None, need: tuple[str, ...]) -> dict[str, Any]:
    if not ref:
        ref = cat["recommended"]["vision" if "vision" in need else "agent"]
        if not ref:
            raise LookupError("no installed model can " + (", ".join(need) or "chat"))
    for m in cat["models"]:
        if m["ref"] == ref:
            missing = [c for c in need if not m["caps"].get(c)]
            if missing:
                raise ValueError(f"{m['name']} cannot do {', '.join(missing)}")
            return m
    if ref and ref.startswith("gguf:"):
        path = Path(ref[5:])
        if path.is_file():  # a file chosen outside the scanned folders
            meta = read_gguf_metadata(path)
            return {"ref": ref, "provider": "llama.cpp", "name": path.stem, "label": path.stem,
                    "path": str(path), "mmproj": None, "context": meta.get("context_length"),
                    "caps": {"tools": meta["tools"], "vision": False, "thinking": meta["thinking"],
                             "audio": False}}
    raise LookupError(f"model {ref!r} is not installed or not reachable")
