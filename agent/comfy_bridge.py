"""Read-only introspection of the running ComfyUI, in-process.

The agent runs inside the ComfyUI server, so it reads node schemas, model
lists, the queue and history directly instead of fetching the ~3000-node
/object_info over HTTP. Schemas are built exactly the way ComfyUI's own
/object_info builds them, including V3-schema nodes (GET_NODE_INFO_V1).

Nothing here changes anything. API nodes (which call paid cloud services) are
flagged and hidden from search by default: this agent is local-first.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .config import COMFY_ROOT

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
_CACHE: dict[str, Any] = {}
_LOCK = threading.Lock()

# Tests replace this with a fixture; inside ComfyUI it reads the live mappings.
_source: Callable[[], dict[str, dict[str, Any]]] | None = None


def set_source(fn: Callable[[], dict[str, dict[str, Any]]] | None) -> None:
    global _source
    _source = fn
    _CACHE.clear()


def _live_object_info() -> dict[str, dict[str, Any]]:
    import nodes  # type: ignore

    out: dict[str, dict[str, Any]] = {}
    for name, cls in list(nodes.NODE_CLASS_MAPPINGS.items()):
        try:
            if hasattr(cls, "GET_NODE_INFO_V1"):
                info = cls.GET_NODE_INFO_V1()
            else:
                inputs = cls.INPUT_TYPES()
                info = {
                    "input": inputs,
                    "input_order": {k: list(v.keys()) for k, v in inputs.items()},
                    "output": list(cls.RETURN_TYPES),
                    "output_name": list(getattr(cls, "RETURN_NAMES", cls.RETURN_TYPES)),
                    "name": name,
                    "display_name": nodes.NODE_DISPLAY_NAME_MAPPINGS.get(name, name),
                    "description": getattr(cls, "DESCRIPTION", "") or "",
                    "category": getattr(cls, "CATEGORY", "sd"),
                    "output_node": bool(getattr(cls, "OUTPUT_NODE", False)),
                    "python_module": getattr(cls, "RELATIVE_PYTHON_MODULE", "nodes"),
                }
                if hasattr(cls, "API_NODE"):
                    info["api_node"] = cls.API_NODE
                if getattr(cls, "DEPRECATED", False):
                    info["deprecated"] = True
            out[name] = info
        except Exception as exc:  # noqa: BLE001  (a broken custom node must not break the agent)
            out[name] = {"name": name, "broken": str(exc)[:200], "input": {}, "output": [],
                         "output_name": [], "category": "broken"}
    return out


def object_info(refresh: bool = False) -> dict[str, dict[str, Any]]:
    with _LOCK:
        if refresh or "object_info" not in _CACHE or time.time() - _CACHE.get("at", 0) > 300:
            _CACHE["object_info"] = (_source or _live_object_info)()
            _CACHE["at"] = time.time()
            _CACHE.pop("index", None)
        return _CACHE["object_info"]


# --- schema normalisation ---------------------------------------------------


def parse_input(name: str, spec: Any) -> dict[str, Any]:
    """Normalise one input spec into {name, type, widget, options...}."""
    if not isinstance(spec, (list, tuple)) or not spec:
        return {"name": name, "type": "*", "widget": False}
    kind, opts = spec[0], (spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {})
    entry: dict[str, Any] = {"name": name}
    if isinstance(kind, list):                    # legacy combo: the list IS the type
        entry.update(type="COMBO", options=[str(o) for o in kind])
    elif kind == "COMBO":                          # V3 combo
        entry.update(type="COMBO", options=[str(o) for o in opts.get("options", [])])
    else:
        entry["type"] = str(kind)
    entry["widget"] = entry["type"] in WIDGET_TYPES and not opts.get("forceInput")
    for key in ("default", "min", "max", "step", "multiline", "tooltip"):
        if key in opts:
            entry[key] = opts[key]
    if opts.get("control_after_generate"):
        entry["seed_like"] = True
    return entry


def schema(node_type: str) -> dict[str, Any] | None:
    info = object_info().get(node_type)
    if info is None:
        return None
    inputs = []
    for group in ("required", "optional"):
        for name, spec in (info.get("input", {}).get(group) or {}).items():
            entry = parse_input(name, spec)
            entry["required"] = group == "required"
            inputs.append(entry)
    outputs = [{"index": i, "type": str(t), "name": str(n)}
               for i, (t, n) in enumerate(zip(info.get("output", []),
                                              info.get("output_name") or info.get("output", [])))]
    return {
        "type": node_type,
        # V3-schema nodes may report display_name as None
        "display": info.get("display_name") or node_type,
        "category": info.get("category", ""),
        "description": (info.get("description") or "")[:400],
        "inputs": inputs,
        "outputs": outputs,
        "output_node": bool(info.get("output_node")),
        "api_node": bool(info.get("api_node")),
        "deprecated": bool(info.get("deprecated")),
        "broken": info.get("broken"),
    }


def compact_schema(node_type: str, max_options: int = 12) -> str:
    """A token-cheap one-node description for the model."""
    s = schema(node_type)
    if s is None:
        return f"{node_type}: UNKNOWN node type"
    links = [f"{i['name']}:{i['type']}{'' if i['required'] else '?'}"
             for i in s["inputs"] if not i["widget"]]
    widgets = []
    for i in s["inputs"]:
        if not i["widget"]:
            continue
        if i["type"] == "COMBO":
            opts = i.get("options", [])
            shown = ",".join(opts[:max_options]) + (f",…+{len(opts) - max_options}" if len(opts) > max_options else "")
            widgets.append(f"{i['name']}:[{shown}]")
        else:
            bounds = ""
            if "min" in i or "max" in i:
                bounds = f"({i.get('min', '')}..{i.get('max', '')})"
            default = f"={i['default']}" if "default" in i and i["type"] != "STRING" else ""
            widgets.append(f"{i['name']}:{i['type']}{bounds}{default}")
    outs = ", ".join(f"{o['index']}:{o['name']}({o['type']})" for o in s["outputs"])
    flags = []
    if s["output_node"]:
        flags.append("OUTPUT NODE")
    if s["api_node"]:
        flags.append("PAID API NODE")
    if s["deprecated"]:
        flags.append("deprecated")
    head = f"{node_type}" + (f" [{s['display']}]" if s["display"] != node_type else "")
    return (f"{head} ({s['category']}){' ' + ' '.join(flags) if flags else ''}\n"
            f"  links in: {', '.join(links) or '-'}\n"
            f"  widgets: {', '.join(widgets) or '-'}\n"
            f"  outputs: {outs or '-'}")


# --- search -----------------------------------------------------------------

_SYNONYMS = {
    "upscale": ["upscale", "esrgan", "resize"], "load image": ["loadimage"],
    "text": ["cliptextencode", "prompt"], "prompt": ["cliptextencode", "prompt"],
    "sampler": ["ksampler", "sampler"], "save": ["saveimage", "save"],
    "lora": ["loraloader", "lora"], "controlnet": ["controlnet"], "mask": ["mask"],
    "video": ["video", "vhs"], "face": ["face"], "inpaint": ["inpaint"],
}


def _index() -> list[tuple[str, str, dict[str, Any]]]:
    if "index" not in _CACHE:
        rows = []
        for name, info in object_info().items():
            hay = " ".join([name, str(info.get("display_name", "")), str(info.get("category", "")),
                            str(info.get("description", ""))[:300],
                            " ".join(map(str, info.get("search_aliases") or []))]).lower()
            rows.append((name, hay, info))
        _CACHE["index"] = rows
    return _CACHE["index"]


def search_nodes(query: str, limit: int = 12, include_api: bool = False) -> list[dict[str, Any]]:
    words = [w for w in re.split(r"[^a-z0-9]+", (query or "").lower()) if w]
    expanded = set(words)
    for w in words:
        expanded.update(_SYNONYMS.get(w, []))
    scored = []
    for name, hay, info in _index():
        if info.get("api_node") and not include_api:
            continue
        if info.get("broken") or info.get("deprecated"):
            continue
        lname = name.lower()
        score = 0.0
        for w in expanded:
            if w == lname:
                score += 12
            elif lname.startswith(w):
                score += 6
            elif w in lname:
                score += 4
            elif w in hay:
                score += 1.5
        if not score:
            continue
        # core ComfyUI nodes are the most reliable building blocks
        if info.get("python_module", "nodes") in ("nodes", "comfy_extras") or \
                str(info.get("python_module", "")).startswith("comfy_extras"):
            score += 2
        scored.append((score, name, info))
    scored.sort(key=lambda r: (-r[0], len(r[1])))
    return [{"type": name, "display": info.get("display_name") or name,
             "category": info.get("category", ""),
             "description": str(info.get("description", ""))[:160]}
            for _, name, info in scored[:limit]]


# --- models -----------------------------------------------------------------


def model_folders() -> list[str]:
    try:
        import folder_paths  # type: ignore

        return sorted(folder_paths.folder_names_and_paths.keys())
    except Exception:  # noqa: BLE001
        return []


def list_models(folder: str, query: str = "", limit: int = 60) -> dict[str, Any]:
    import folder_paths  # type: ignore

    if folder not in folder_paths.folder_names_and_paths:
        return {"error": f"unknown model folder {folder!r}", "folders": model_folders()}
    files = folder_paths.get_filename_list(folder)
    q = (query or "").lower()
    hits = [f for f in files if q in f.lower()] if q else list(files)
    return {"folder": folder, "total": len(files), "matched": len(hits), "files": hits[:limit],
            "truncated": len(hits) > limit}


# --- queue, history, outputs -------------------------------------------------


def _server():
    from server import PromptServer  # type: ignore

    return PromptServer.instance


def queue_status() -> dict[str, Any]:
    running, pending = _server().prompt_queue.get_current_queue()
    return {"running": [r[1] for r in running], "pending": [p[1] for p in pending],
            "remaining": len(running) + len(pending)}


def recent_runs(limit: int = 5) -> list[dict[str, Any]]:
    history = _server().prompt_queue.get_history(max_items=limit)
    runs = []
    for prompt_id, entry in list(history.items())[-limit:]:
        status = entry.get("status") or {}
        error = None
        for kind, data in status.get("messages") or []:
            if kind == "execution_error":
                error = {"node_id": data.get("node_id"), "node_type": data.get("node_type"),
                         "message": str(data.get("exception_message", ""))[:600],
                         "exception": data.get("exception_type")}
        images = []
        for node_id, out in (entry.get("outputs") or {}).items():
            for img in out.get("images") or []:
                images.append({"node_id": node_id, "filename": img.get("filename"),
                               "subfolder": img.get("subfolder", ""), "type": img.get("type", "output")})
        runs.append({"prompt_id": prompt_id, "status": status.get("status_str"),
                     "completed": status.get("completed"), "error": error, "images": images[-8:]})
    return list(reversed(runs))


def output_image_path(image: dict[str, Any]) -> Path:
    import folder_paths  # type: ignore

    base = Path(folder_paths.get_output_directory() if image.get("type", "output") == "output"
                else folder_paths.get_temp_directory() if image.get("type") == "temp"
                else folder_paths.get_input_directory())
    target = (base / (image.get("subfolder") or "") / str(image.get("filename", ""))).resolve()
    if base.resolve() not in target.parents:
        raise ValueError("image path escapes its ComfyUI folder")
    return target


def system_stats() -> dict[str, Any]:
    try:
        import comfy.model_management as mm  # type: ignore

        device = mm.get_torch_device()
        total = mm.get_total_memory(device)
        free = mm.get_free_memory(device)
        return {"device": str(device), "vram_total_gb": round(total / 1e9, 1),
                "vram_free_gb": round(free / 1e9, 1)}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


# --- saved workflows ---------------------------------------------------------


def workflows_dir() -> Path:
    return COMFY_ROOT / "user" / "default" / "workflows"


def list_workflows(query: str = "", limit: int = 30) -> list[str]:
    root = workflows_dir()
    if not root.is_dir():
        return []
    q = (query or "").lower()
    names = sorted(str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*.json"))
    words = [w for w in q.split() if w]
    hits = [n for n in names if all(w in n.lower() for w in words)] if words else names
    return hits[:limit]


def read_workflow(name: str) -> dict[str, Any]:
    root = workflows_dir().resolve()
    target = (root / name).resolve()
    if root not in target.parents or target.suffix != ".json" or not target.is_file():
        raise ValueError(f"no saved workflow named {name!r}")
    return json.loads(target.read_text(encoding="utf-8"))
