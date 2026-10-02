"""HTTP routes for the agent sidebar, attached to ComfyUI's own PromptServer.

No new listener and no CORS changes: these routes inherit ComfyUI's local
server. Agent runs execute on a worker thread and stream events over
ComfyUI's existing websocket (event "ezeai.agent"), so tokens appear as they
are generated without a second connection.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from typing import Any

from .agent import config, graph_ops, memory, models, providers, recipes, runtime, sessions
from .agent.loop import Agent

PREFIX = "/ezeai/agent"
API_VERSION = "1.0.0"
MAX_BODY = 24 * 1024 * 1024      # snapshots of big graphs plus a few images
MAX_IMAGES = 4

_RUNS: dict[str, dict[str, Any]] = {}
_RUNS_LOCK = threading.Lock()


def _server():
    from server import PromptServer  # type: ignore

    return PromptServer.instance


class _Emitter:
    """Batches token deltas (~16 messages/s) and forwards everything else at once."""

    def __init__(self, run_id: str, client_id: str | None):
        self.run_id, self.client_id = run_id, client_id
        self.buffer: dict[str, list[str]] = {}
        self.last_flush = time.time()
        self.lock = threading.Lock()

    def _send(self, payload: dict[str, Any]) -> None:
        try:
            _server().send_sync("ezeai.agent", {"run_id": self.run_id, **payload}, self.client_id)
        except Exception:  # noqa: BLE001  (a closed tab must not crash the run)
            pass

    def flush(self) -> None:
        with self.lock:
            buffered, self.buffer = self.buffer, {}
            self.last_flush = time.time()
        for kind, parts in buffered.items():
            if parts:
                self._send({"type": "delta", "kind": kind, "text": "".join(parts)})

    def __call__(self, event: dict[str, Any]) -> None:
        if event.get("type") == "delta":
            with self.lock:
                self.buffer.setdefault(event["kind"], []).append(event["text"])
            if time.time() - self.last_flush > 0.06:
                self.flush()
            return
        self.flush()
        self._send(event)


def _run_agent(run_id: str, chat_id: str, text: str, snapshot: dict | None,
               images: list[str], client_id: str | None, canvas_image: str | None = None) -> None:
    emit = _Emitter(run_id, client_id)
    cancel = _RUNS[run_id]["cancel"]
    chat = sessions.load(chat_id)
    emit({"type": "start", "chat_id": chat_id})
    try:
        result = Agent(emit, cancel).run(chat, text, snapshot, images, canvas_image)
        chat["turns"].append({"user": text, "assistant": result["text"],
                              "proposals": result["proposals"], "model": result["model"],
                              "at": int(time.time()), "rounds": result["rounds"],
                              "usage": result["usage"], "elapsed": result["elapsed"],
                              "trace": result["trace"], "images": len(images)})
        if not chat.get("title"):
            chat["title"] = " ".join(text.split())[:60]
        sessions.save(chat)
        emit({"type": "done", "result": result, "chat_id": chat_id})
    except providers.Cancelled:
        emit({"type": "stopped", "chat_id": chat_id,
              "reason": "time limit reached" if time.time() - _RUNS[run_id]["started"]
              >= config.load_settings()["run_timeout_s"] - 1 else "stopped by you"})
    except Exception as exc:  # noqa: BLE001
        emit({"type": "error", "message": f"{type(exc).__name__}: {exc}",
              "detail": traceback.format_exc(limit=3)})
    finally:
        emit.flush()
        with _RUNS_LOCK:
            _RUNS.pop(run_id, None)


# --- handlers (plain functions so they are testable without aiohttp) -------


def status() -> dict[str, Any]:
    cat = models.catalogue()
    settings = config.load_settings()
    try:
        available = recipes.available()
    except Exception:  # noqa: BLE001
        available = {}
    return {
        "ok": True, "api_version": API_VERSION, "settings": settings,
        "recommended": cat["recommended"], "model_count": len(cat["models"]),
        "ollama_error": cat["ollama_error"], "llama_server": cat["llama_server"],
        "runtime": runtime.status(), "recipes": available,
        "skills": sorted(memory.skills()), "memory_notes": len(memory.notes()),
        "active_runs": len(_RUNS),
        "runs": [{"run_id": k, "chat_id": v["chat_id"], "age_s": round(time.time() - v["started"])}
                 for k, v in list(_RUNS.items())],
    }


def send(body: dict[str, Any]) -> dict[str, Any]:
    text = str(body.get("text") or "").strip()
    if not text:
        return {"ok": False, "error": "empty message"}
    chat_id = str(body.get("chat_id") or sessions.new_id())
    sessions.load(chat_id)  # validates the id
    images = [str(i) for i in (body.get("images") or [])][:MAX_IMAGES]
    with _RUNS_LOCK:
        if any(r["chat_id"] == chat_id for r in _RUNS.values()):
            return {"ok": False, "error": "this chat is already answering; stop it or wait"}
        run_id = "r_" + uuid.uuid4().hex[:10]
        _RUNS[run_id] = {"chat_id": chat_id, "cancel": threading.Event(), "started": time.time()}
    threading.Thread(target=_run_agent, name=f"ezeai-agent-{run_id}", daemon=True,
                     args=(run_id, chat_id, text, body.get("snapshot"), images,
                           body.get("client_id"), body.get("canvas_image") or None)).start()
    return {"ok": True, "run_id": run_id, "chat_id": chat_id}


def stop(body: dict[str, Any]) -> dict[str, Any]:
    if body.get("all"):
        for run in list(_RUNS.values()):
            run["cancel"].set()
        return {"ok": True, "stopped": len(_RUNS)}
    run = _RUNS.get(str(body.get("run_id") or ""))
    if run is None:
        return {"ok": False, "error": "no such run (it may have finished)"}
    run["cancel"].set()
    return {"ok": True}


def revalidate(body: dict[str, Any]) -> dict[str, Any]:
    """Check a card's ops against the canvas as it is NOW, just before applying.

    The canvas may have changed since the agent proposed the card. Applying a
    stale plan blindly could wire into a node the user has since deleted.
    """
    ops = body.get("ops") or []
    # cards carry resolved ops (virtual ids "v1"... and existing ids); re-plan
    # from their original form
    raw = []
    for op in ops:
        op = dict(op)
        if op.get("op") == "add_node":
            raw.append({"op": "add_node", "ref": op.get("vid") or op.get("ref"), "type": op["type"],
                        "widgets": op.get("widgets") or {}, "title": op.get("title")})
        else:
            raw.append(op)
    result = graph_ops.plan(raw, body.get("snapshot"))
    result.pop("graph", None)
    return {"ok": result["ok"], "errors": result["errors"], "warnings": result["warnings"],
            "risk": result["risk"]}


def save_workflow(body: dict[str, Any]) -> dict[str, Any]:
    """Write the canvas into the user's workflow library.

    A new file is written directly. An existing one is only replaced when the
    request says overwrite=true, which the UI sends only after the user clicked.
    """
    import re as _re

    from .agent import comfy_bridge

    name = _re.sub(r"[^A-Za-z0-9 ._()-]", "", str(body.get("name", ""))).strip()[:84]
    if not name.lower().endswith(".json"):
        name += ".json"
    root = comfy_bridge.workflows_dir().resolve()
    target = (root / name).resolve()
    if target.parent != root:
        return {"ok": False, "error": "bad workflow name"}
    if target.exists() and not body.get("overwrite"):
        return {"ok": False, "error": f"{name} already exists", "exists": True}
    graph = body.get("graph")
    if not isinstance(graph, dict) or "nodes" not in graph:
        return {"ok": False, "error": "no graph to save"}
    config.atomic_write_json(target, graph)
    return {"ok": True, "name": name, "path": str(target)}


def prepare_queue() -> dict[str, Any]:
    """Free the GPU before a render: the LLM and the diffusion model compete."""
    if not config.load_settings().get("free_vram_before_queue", True):
        return {"ok": True, "freed": None}
    return {"ok": True, "freed": providers.unload()}


def register_routes(server) -> bool:
    try:
        from aiohttp import web
    except Exception:  # noqa: BLE001
        return False
    routes = server.routes

    async def body_of(request):
        raw = await request.content.read(MAX_BODY + 1)
        if len(raw) > MAX_BODY:
            raise ValueError("request too large")
        return json.loads(raw.decode("utf-8")) if raw else {}

    def handler(fn, takes_body=True):
        async def wrapped(request):
            try:
                import asyncio

                loop = asyncio.get_running_loop()
                if takes_body:
                    body = await body_of(request)
                    data = await loop.run_in_executor(None, fn, body)
                else:
                    data = await loop.run_in_executor(None, fn)
                return web.json_response(data)
            except Exception as exc:  # noqa: BLE001
                return web.json_response({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return wrapped

    routes.get(f"{PREFIX}/status")(handler(status, takes_body=False))
    routes.get(f"{PREFIX}/models")(handler(lambda: models.catalogue(refresh=True), takes_body=False))
    routes.post(f"{PREFIX}/settings")(handler(lambda b: {"ok": True, "settings": config.save_settings(b)}))
    routes.get(f"{PREFIX}/chats")(handler(lambda: {"ok": True, "chats": sessions.listing()}, takes_body=False))
    routes.post(f"{PREFIX}/chat")(handler(lambda b: {"ok": True, "chat": sessions.load(str(b.get("id")))}))
    routes.post(f"{PREFIX}/chat/delete")(handler(lambda b: {"ok": sessions.delete(str(b.get("id")))}))
    routes.post(f"{PREFIX}/send")(handler(send))
    routes.post(f"{PREFIX}/stop")(handler(stop))
    routes.post(f"{PREFIX}/revalidate")(handler(revalidate))
    routes.post(f"{PREFIX}/prepare_queue")(handler(lambda b: prepare_queue()))
    routes.post(f"{PREFIX}/save_workflow")(handler(save_workflow))
    routes.post(f"{PREFIX}/unload")(handler(lambda b: {"ok": True, "freed": providers.unload()}))
    routes.get(f"{PREFIX}/memory")(handler(lambda: {"ok": True, "notes": memory.notes()}, takes_body=False))
    routes.post(f"{PREFIX}/memory")(handler(lambda b: memory.remember(str(b.get("note", "")),
                                                                       bool(b.get("forget")))))
    return True
