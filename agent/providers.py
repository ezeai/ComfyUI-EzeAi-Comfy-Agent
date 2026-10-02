"""One chat interface over every local runtime.

`chat()` takes Ollama-style messages and returns one normalized reply:

    {"content", "thinking", "tool_calls": [{"id", "name", "arguments": dict}],
     "usage": {"prompt", "completion"}, "elapsed", "model", "provider"}

The differences it hides are real and were measured on this machine: Ollama
returns tool arguments as an object, llama-server returns them as a JSON
string; Ollama streams newline-delimited JSON, llama-server streams SSE with
tool calls split across chunks.

Local-first: a model server that is not on this PC is refused unless the user
explicitly allows remote hosts in Settings.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

from . import runtime
from .config import load_settings

Delta = Callable[[str, str], None]   # (kind: "content" | "thinking", text)

LOOPBACK = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


class ProviderError(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


class Degenerate(ProviderError):
    """The model fell into repeating itself; its output was cut off."""


def _repeating(text: str, min_repeats: int = 6) -> bool:
    """True when the output ends in one unit repeated back-to-back.

    Small local models sometimes loop on a phrase forever. A degenerate loop
    repeats EXACTLY and consecutively; a legitimate list ("node 3 ..., node 5
    ...") varies between items, so it is not flagged. Measured: an earlier
    "count occurrences" version wrongly flagged list-like prose.
    """
    tail = text[-1200:]
    for unit in range(8, 160):
        if unit * min_repeats > len(tail):
            break
        piece = tail[-unit:]
        if piece.strip() and tail.endswith(piece * min_repeats):
            return True
    return False


def _check_local(url: str) -> None:
    host = (urlparse(url).hostname or "").lower()
    if host in LOOPBACK or host.endswith(".localhost"):
        return
    if load_settings().get("allow_remote_hosts"):
        return
    raise ProviderError(f"refusing to send your data to {host}: EzeAi Comfy Agent is local-first. "
                        "Enable 'allow remote hosts' in Settings only if you trust that machine.")


def _stream_lines(url: str, body: dict, headers: dict, cancel: threading.Event | None,
                  timeout: float = 600) -> Iterable[bytes]:
    _check_local(url)
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json", **headers})
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:600]
        raise ProviderError(f"model server returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"model server not reachable at {url}: {exc.reason}") from exc
    with response:
        for line in response:
            if cancel is not None and cancel.is_set():
                raise Cancelled("stopped by the user")
            line = line.strip()
            if line:
                yield line


# --- message conversion -----------------------------------------------------


def _to_openai(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        role = m["role"]
        if role == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("tool_call_id") or "call",
                        "content": m.get("content", "")})
            continue
        entry: dict[str, Any] = {"role": role}
        images = m.get("images") or []
        if images:
            parts: list[dict] = [{"type": "text", "text": m.get("content", "")}]
            for b64 in images:
                parts.append({"type": "image_url",
                              "image_url": {"url": f"data:image/png;base64,{b64}"}})
            entry["content"] = parts
        else:
            entry["content"] = m.get("content", "")
        if m.get("tool_calls"):
            entry["tool_calls"] = [{
                "id": c.get("id") or f"call_{i}", "type": "function",
                "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments") or {})},
            } for i, c in enumerate(m["tool_calls"])]
        out.append(entry)
    return out


def _to_ollama(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        entry: dict[str, Any] = {"role": m["role"], "content": m.get("content", "")}
        if m.get("images"):
            entry["images"] = m["images"]
        if m.get("tool_calls"):
            entry["tool_calls"] = [{"function": {"name": c["name"],
                                                 "arguments": c.get("arguments") or {}}}
                                   for c in m["tool_calls"]]
        if m["role"] == "tool" and m.get("tool_name"):
            entry["tool_name"] = m["tool_name"]
        out.append(entry)
    return out


def _parse_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except (TypeError, ValueError):
        return {"_unparsed": str(raw)}


# --- text tool-call fallback ------------------------------------------------
# Some models answer tool requests with the call written into the text instead
# of the structured field (and models with no tool training need it). Both are
# recovered here rather than silently treated as a final answer.

_TAGGED = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_FENCED = re.compile(r"```(?:json|tool_call)?\s*(\{\s*\"name\".*?\})\s*```", re.S)


def parse_text_tool_calls(content: str, known: set[str]) -> tuple[list[dict], str]:
    calls, spans = [], []
    for pattern in (_TAGGED, _FENCED):
        for match in pattern.finditer(content or ""):
            try:
                data = json.loads(match.group(1))
            except ValueError:
                continue
            name = data.get("name") or (data.get("function") or {}).get("name")
            if name not in known:
                continue
            args = data.get("arguments", data.get("parameters", {}))
            calls.append({"id": f"txt_{uuid.uuid4().hex[:8]}", "name": name,
                          "arguments": _parse_args(args)})
            spans.append(match.span())
    if not calls:
        return [], content
    cleaned = content
    for start, end in sorted(spans, reverse=True):
        cleaned = cleaned[:start] + cleaned[end:]
    return calls, cleaned.strip()


def text_tool_protocol(tools: list[dict]) -> str:
    lines = ["You can call tools. To call one, reply with exactly:",
             '<tool_call>{"name": "<tool name>", "arguments": {...}}</tool_call>',
             "One block per call. After the results arrive, continue. Tools:"]
    for t in tools:
        f = t["function"]
        params = json.dumps(f.get("parameters", {}).get("properties", {}))[:600]
        lines.append(f"- {f['name']}: {f['description'][:300]} params={params}")
    return "\n".join(lines)


# --- providers --------------------------------------------------------------


def _ollama_chat(model: dict, messages, tools, *, on_delta, cancel, options) -> dict:
    settings = load_settings()
    base = settings["ollama_url"].rstrip("/")
    body: dict[str, Any] = {
        "model": model["name"], "messages": _to_ollama(messages), "stream": True,
        "keep_alive": options.get("keep_alive", settings["keep_alive"]),
        "options": {"temperature": options.get("temperature", settings["temperature"]),
                    "num_ctx": int(options.get("context", settings["context_tokens"]))},
    }
    if options.get("seed") is not None:
        body["options"]["seed"] = int(options["seed"])
    body["options"]["num_predict"] = int(options.get("max_tokens", settings["max_tokens"]))
    if model["caps"].get("thinking"):
        body["think"] = bool(options.get("think", settings["think"]))
    if tools and model["caps"].get("tools"):
        body["tools"] = tools
    if options.get("format"):
        body["format"] = options["format"]

    content, thinking, calls, usage = [], [], [], {}
    for line in _stream_lines(f"{base}/api/chat", body, {}, cancel):
        chunk = json.loads(line)
        if chunk.get("error"):
            raise ProviderError(chunk["error"])
        msg = chunk.get("message") or {}
        if msg.get("thinking"):
            thinking.append(msg["thinking"])
            on_delta("thinking", msg["thinking"])
        if msg.get("content"):
            content.append(msg["content"])
            on_delta("content", msg["content"])
            if len(content) % 40 == 0 and _repeating("".join(content)):
                raise Degenerate("the model started repeating itself, so its answer was cut off")
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            calls.append({"id": call.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                          "name": fn.get("name", ""), "arguments": _parse_args(fn.get("arguments"))})
        if chunk.get("done"):
            usage = {"prompt": chunk.get("prompt_eval_count"),
                     "completion": chunk.get("eval_count")}
    return {"content": "".join(content), "thinking": "".join(thinking),
            "tool_calls": calls, "usage": usage}


def _openai_chat(url: str, model_id: str, headers: dict, messages, tools, *, model,
                 on_delta, cancel, options) -> dict:
    settings = load_settings()
    body: dict[str, Any] = {
        "model": model_id, "messages": _to_openai(messages), "stream": True,
        "temperature": options.get("temperature", settings["temperature"]),
        "stream_options": {"include_usage": True},
    }
    if options.get("seed") is not None:
        body["seed"] = int(options["seed"])
    body["max_tokens"] = int(options.get("max_tokens", settings["max_tokens"]))
    if tools and model["caps"].get("tools"):
        body["tools"] = tools
    if options.get("format"):
        body["response_format"] = {"type": "json_object"}

    content, thinking, usage = [], [], {}
    pending: dict[int, dict] = {}
    for line in _stream_lines(f"{url}/v1/chat/completions", body, headers, cancel):
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            break
        chunk = json.loads(payload)
        if chunk.get("usage"):
            usage = {"prompt": chunk["usage"].get("prompt_tokens"),
                     "completion": chunk["usage"].get("completion_tokens")}
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                thinking.append(delta["reasoning_content"])
                on_delta("thinking", delta["reasoning_content"])
            if delta.get("content"):
                content.append(delta["content"])
                on_delta("content", delta["content"])
                if len(content) % 40 == 0 and _repeating("".join(content)):
                    raise Degenerate("the model started repeating itself, so its answer was cut off")
            for part in delta.get("tool_calls") or []:
                slot = pending.setdefault(part.get("index", 0), {"id": None, "name": "", "args": ""})
                if part.get("id"):
                    slot["id"] = part["id"]
                fn = part.get("function") or {}
                slot["name"] += fn.get("name") or ""
                slot["args"] += fn.get("arguments") or ""
    calls = [{"id": s["id"] or f"call_{i}", "name": s["name"], "arguments": _parse_args(s["args"])}
             for i, s in sorted(pending.items())]
    return {"content": "".join(content), "thinking": "".join(thinking),
            "tool_calls": calls, "usage": usage}


def chat(model: dict, messages: list[dict], tools: list[dict] | None = None, *,
         on_delta: Delta | None = None, cancel: threading.Event | None = None,
         options: dict | None = None) -> dict:
    """Run one model turn. Raises ProviderError / Cancelled."""
    options = dict(options or {})
    on_delta = on_delta or (lambda kind, text: None)
    tools = tools or []
    known = {t["function"]["name"] for t in tools}

    # A model without native tool support gets the text protocol instead of
    # silently ignoring the tools.
    if tools and not model["caps"].get("tools"):
        messages = [dict(m) for m in messages]
        protocol = text_tool_protocol(tools)
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = messages[0]["content"] + "\n\n" + protocol
        else:
            messages.insert(0, {"role": "system", "content": protocol})

    started = time.time()
    provider = model["provider"]
    if provider == "ollama":
        result = _ollama_chat(model, messages, tools, on_delta=on_delta, cancel=cancel,
                              options=options)
    elif provider == "llama.cpp":
        run = runtime.ensure(model, context=int(options.get("context",
                                                            load_settings()["context_tokens"])))
        result = _openai_chat(f"http://127.0.0.1:{run['port']}", "local",
                              {"Authorization": f"Bearer {run['api_key']}"}, messages, tools,
                              model=model, on_delta=on_delta, cancel=cancel, options=options)
        runtime.schedule_stop(runtime.keep_alive_seconds(
            options.get("keep_alive", load_settings()["keep_alive"])))
    elif provider == "openai-compatible":
        url = load_settings()["external_llama_url"].rstrip("/")
        result = _openai_chat(url, model["name"], {}, messages, tools, model=model,
                              on_delta=on_delta, cancel=cancel, options=options)
    else:
        raise ProviderError(f"unknown provider {provider!r}")

    if not result["tool_calls"] and known:
        recovered, cleaned = parse_text_tool_calls(result["content"], known)
        if recovered:
            result["tool_calls"] = recovered
            result["content"] = cleaned
            result["recovered_text_tool_calls"] = True
    result["elapsed"] = round(time.time() - started, 2)
    result["model"] = model["ref"]
    result["provider"] = provider
    return result


def unload(model: dict | None = None) -> dict[str, Any]:
    """Free the GPU before a render: stop llama-server and ask Ollama to unload."""
    freed = {"llama_server": runtime.stop("render queued"), "ollama": []}
    base = load_settings()["ollama_url"].rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/ps", timeout=4) as r:
            loaded = json.loads(r.read().decode()).get("models", [])
    except Exception:  # noqa: BLE001
        loaded = []
    for entry in loaded:
        name = entry.get("name")
        if model and model.get("provider") == "ollama" and model.get("name") != name:
            continue
        try:
            req = urllib.request.Request(f"{base}/api/generate",
                                         data=json.dumps({"model": name, "keep_alive": 0}).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=20).read()
            freed["ollama"].append(name)
        except Exception:  # noqa: BLE001
            pass
    return freed
