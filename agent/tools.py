"""The agent's tools.

LOOK tools run automatically: they only read (the canvas, node schemas, model
files, the queue, the last run, an output image).

DO tools never touch anything. They validate what the model asked for and turn
it into a PROPOSAL - a card the user can preview and apply. A proposal's risk
level is computed here from what it actually does, never taken from the model,
and policy.py decides from that risk whether it may auto-apply.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from . import comfy_bridge as bridge
from . import graph_ops, memory, recipes


@dataclass
class ToolContext:
    snapshot: dict[str, Any] | None
    vision: Callable[[str, str], str] | None = None  # (b64 png, question) -> description
    proposals: list[dict[str, Any]] = field(default_factory=list)
    attached_images: list[str] = field(default_factory=list)  # b64 for a vision-capable agent model
    agent_sees: bool = False
    canvas_image: str | None = None   # b64 screenshot of the user's canvas, if shared
    # Flat edit tools accumulate here and are validated together; the whole
    # draft becomes ONE card when the turn ends (see finalize_draft).
    draft_ops: list[dict[str, Any]] = field(default_factory=list)


def _fn(name: str, description: str, properties: dict | None = None,
        required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {},
                       **({"required": required} if required else {})}}}


LOOK = [
    _fn("read_graph", "Read the workflow currently on the canvas: every node with its #id, type, "
                      "widget values and links, plus problems that would make it fail. Call this "
                      "before changing or explaining the graph."),
    _fn("search_nodes", "Find installed node types by what they do (e.g. 'upscale', 'load image', "
                        "'lora', 'face detailer'). Returns type names to use in add_node.",
        {"query": {"type": "string"}, "limit": {"type": "integer", "description": "default 10"}},
        ["query"]),
    _fn("node_info", "Exact inputs, widgets (with allowed values and ranges) and outputs of one "
                     "node type. Call before add_node with a type you have not checked.",
        {"type": {"type": "string", "description": "exact node type, e.g. KSampler"}}, ["type"]),
    _fn("list_models", "Model files installed in a ComfyUI models folder. Use exact filenames from "
                       "here in loader widgets; never invent one.",
        {"folder": {"type": "string", "description": "checkpoints, loras, vae, diffusion_models, "
                                                     "text_encoders, upscale_models, controlnet, clip_vision..."},
         "query": {"type": "string", "description": "optional words to filter by"}}, ["folder"]),
    _fn("list_workflows", "The user's saved workflows, optionally filtered by words.",
        {"query": {"type": "string"}}),
    _fn("check_last_run", "The queue, and the most recent runs: success or the exact error "
                          "(node type + message) and the images they produced."),
    _fn("look_at_output", "See an output image to judge it (subject, quality, problems). "
                          "Use before saying a result is good.",
        {"filename": {"type": "string", "description": "a filename from check_last_run, or 'latest'"},
         "question": {"type": "string", "description": "what to look for, optional"}}),
    _fn("system_status", "GPU memory, the queue, and which language model is loaded."),
    _fn("look_at_canvas", "SEE the ComfyUI canvas as the user sees it (a screenshot): layout, "
                          "groups, notes, previews, red error outlines. Use when the user says "
                          "'look', 'see', 'on my screen', or the layout matters.",
        {"question": {"type": "string", "description": "what to look for, optional"}}),
    _fn("use_skill", "Load a playbook (see SKILLS in your instructions) before a task it covers.",
        {"name": {"type": "string"}}, ["name"]),
    _fn("remember", "Save a lasting preference the user states (style, model, aspect, platform). "
                    "Only when they state one.",
        {"note": {"type": "string"}, "forget": {"type": "boolean"}}, ["note"]),
]

_NODE = {"type": "string", "description": "an existing node #id from read_graph (e.g. \"3\"), "
                                        "or a ref you gave in add_node"}

EDIT = [
    _fn("insert_after", "Insert a pass-through node right after another node and rewire everything "
                        "that used its outputs. Use for a LoRA after the checkpoint/model loader, a "
                        "model patch after a model loader, or an image filter after an image. Example: "
                        "node \"4\", type LoraLoader, widgets {lora_name, strength_model}.",
        {"node": _NODE, "type": {"type": "string", "description": "node type to insert"},
         "ref": {"type": "string", "description": "short name for the new node"},
         "widgets": {"type": "object", "description": "widget values for the new node"}},
        ["node", "type"]),
    _fn("set_widgets", "Change one or more widget values on a node, e.g. node \"3\", "
                       "values {\"steps\": 30, \"cfg\": 6}.",
        {"node": _NODE, "values": {"type": "object", "description": "widget name -> new value"}},
        ["node", "values"]),
    _fn("add_node", "Add a node. Check its exact widget names with node_info first.",
        {"type": {"type": "string", "description": "exact node type, e.g. LoraLoader"},
         "ref": {"type": "string", "description": "a short name to refer to it in connect, e.g. \"lora\""},
         "widgets": {"type": "object", "description": "widget name -> value, optional"}},
        ["type", "ref"]),
    _fn("connect", "Connect an output of one node to an input of another.",
        {"from_node": _NODE,
         "from_output": {"type": "string", "description": "output name or index, e.g. MODEL or 0"},
         "to_node": _NODE, "to_input": {"type": "string", "description": "input name, e.g. model"}},
        ["from_node", "from_output", "to_node", "to_input"]),
    _fn("disconnect", "Remove the link going into one input.",
        {"node": _NODE, "input": {"type": "string"}}, ["node", "input"]),
    _fn("remove_node", "Delete a node (removing one that was already on the canvas always asks the user).",
        {"node": _NODE}, ["node"]),
    _fn("bypass_node", "Bypass (skip) or re-enable a node.",
        {"node": _NODE, "on": {"type": "boolean", "description": "true = bypass, false = enable"}},
        ["node", "on"]),
]

DO = [
    _fn("open_workflow", "Propose opening one of the user's saved workflows (replaces the canvas, "
                         "so it always asks).",
        {"name": {"type": "string", "description": "exact name from list_workflows"}}, ["name"]),
    _fn("queue_run", "Propose running the current workflow. The agent's language model is unloaded "
                     "first so the render has the GPU.",
        {"count": {"type": "integer", "description": "how many times, default 1, max 8"}}),
    _fn("save_workflow", "Propose saving the current canvas to the user's workflow library. A new "
                         "name is saved directly; overwriting an existing workflow always asks.",
        {"name": {"type": "string", "description": "file name, e.g. 'Fox portrait Z-Image'"}}, ["name"]),
]

LOOK_NAMES = {t["function"]["name"] for t in LOOK}
EDIT_NAMES = {t["function"]["name"] for t in EDIT}
DO_NAMES = {t["function"]["name"] for t in DO} | EDIT_NAMES


def recipe_tool_names() -> set[str]:
    return {r["tool"] for r in recipes.RECIPES.values()}


def all_tools(mode: str = "ask") -> list[dict[str, Any]]:
    """LOOK + one typed tool per available recipe + flat edit tools + DO.

    Inspect mode gets LOOK tools only: the restriction is in the tool list
    itself, so no model output can produce a change card.
    """
    if mode == "inspect":
        return list(LOOK)
    try:
        recipe_tools = recipes.tool_specs()
    except Exception:  # noqa: BLE001
        recipe_tools = []
    return LOOK + recipe_tools + EDIT + DO


def _json(data: Any, limit: int = 6000) -> str:
    text = json.dumps(data, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…(truncated)"


def _image_b64(path, side: int = 1024) -> str:
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((side, side))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _proposal(ctx: ToolContext, kind: str, summary: str, **payload) -> str:
    card = {"id": f"p{len(ctx.proposals) + 1}", "kind": kind, "summary": summary, **payload}
    ctx.proposals.append(card)
    return _json({"ok": True, "proposal": card["id"], "risk": card.get("risk"),
                  "note": "shown to the user as a card; do not claim it is done"})


def run(name: str, args: dict[str, Any], ctx: ToolContext) -> str:
    """Execute one tool and return text for the model."""
    args = args or {}
    try:
        if name == "read_graph":
            if not ctx.draft_ops:
                return graph_ops.summarize(ctx.snapshot)
            pending = graph_ops.plan(ctx.draft_ops, ctx.snapshot)
            return (graph_ops.summarize(_virtual_snapshot(pending["graph"]))
                    + "\n\nPENDING (not applied yet, will be one card):\n"
                    + "\n".join(pending["diff"]))

        if name == "search_nodes":
            return _json(bridge.search_nodes(str(args.get("query", "")),
                                             limit=max(1, min(int(args.get("limit") or 10), 25))))

        if name == "node_info":
            return bridge.compact_schema(str(args.get("type", "")), max_options=20)

        if name == "list_models":
            return _json(bridge.list_models(str(args.get("folder", "")), str(args.get("query", ""))))

        if name == "list_workflows":
            return _json(bridge.list_workflows(str(args.get("query", ""))))

        if name == "check_last_run":
            return _json({"queue": bridge.queue_status(), "runs": bridge.recent_runs(3)})

        if name == "system_status":
            from . import runtime

            return _json({"gpu": bridge.system_stats(), "queue": bridge.queue_status(),
                          "llama_server": runtime.status()})

        if name == "look_at_output":
            want = str(args.get("filename") or "latest")
            image = None
            for run_ in bridge.recent_runs(5):
                for img in reversed(run_["images"]):
                    if want == "latest" or img["filename"] == want:
                        image = img
                        break
                if image:
                    break
            if image is None:
                return "No output image found. Run the workflow first, or check the filename."
            b64 = _image_b64(bridge.output_image_path(image))
            question = str(args.get("question") or "Describe this image and any visible problems.")
            if ctx.agent_sees:
                ctx.attached_images.append(b64)
                return f"The image {image['filename']} is attached to the next message. {question}"
            if ctx.vision is None:
                return "No vision-capable model is installed, so the image cannot be inspected."
            return f"{image['filename']} (seen by the vision model): " + ctx.vision(b64, question)

        if name == "look_at_canvas":
            if not ctx.canvas_image:
                return ("No canvas view was shared with this message (Settings > share canvas view). "
                        "Use read_graph for the structure.")
            question = str(args.get("question") or "Describe this ComfyUI canvas: the nodes, how they "
                                                    "are laid out and grouped, and anything highlighted "
                                                    "in red or looking broken.")
            if ctx.agent_sees:
                ctx.attached_images.append(ctx.canvas_image)
                return f"The canvas screenshot is attached to the next message. {question}"
            if ctx.vision is None:
                return "No vision-capable model is installed, so the canvas cannot be seen."
            return "Canvas (seen by the vision model): " + ctx.vision(ctx.canvas_image, question)

        if name == "use_skill":
            return _json(memory.load_skill(str(args.get("name", ""))), limit=9000)

        if name == "remember":
            return _json(memory.remember(str(args.get("note", "")), bool(args.get("forget"))))

        # --- recipes: a whole proven workflow, as one card
        if name in recipe_tool_names():
            key = recipes.by_tool(name)
            ops = recipes.build(key, args)
            result = graph_ops.plan(ops, {"nodes": []})
            if not result["ok"]:
                return _json({"ok": False, "errors": result["errors"]})
            return _proposal(ctx, "graph_edit", recipes.RECIPES[key]["label"],
                             ops=result["ops"], diff=result["diff"], warnings=result["warnings"],
                             risk="edit", recipe=key, params=args)

        # --- flat edits: validated against the canvas plus everything drafted so far
        if name == "insert_after":
            new_ops, problems = graph_ops.insert_after_ops(
                ctx.snapshot, ctx.draft_ops, _node_ref(args.get("node")), str(args.get("type", "")),
                str(args.get("ref") or "inserted"), _as_dict(args.get("widgets")))
            if problems:
                return _json({"ok": False, "errors": problems})
        if name in EDIT_NAMES:
            if name != "insert_after":
                new_ops = _edit_ops(name, args)
            trial = graph_ops.plan(ctx.draft_ops + new_ops, ctx.snapshot)
            if not trial["ok"]:
                prior = graph_ops.plan(ctx.draft_ops, ctx.snapshot)["errors"] if ctx.draft_ops else []
                fresh = [e for e in trial["errors"] if e not in prior] or trial["errors"]
                return _json({"ok": False, "errors": fresh, "fix": "correct the call and try again"})
            ctx.draft_ops.extend(new_ops)
            return _json({"ok": True, "pending_changes": trial["diff"][-len(new_ops):],
                          "draft_size": len(ctx.draft_ops),
                          "warnings": trial["warnings"][:4],
                          "note": "added to the pending change; it is shown to the user as one card "
                                  "when you finish. Do not claim it is applied."})

        if name == "open_workflow":
            wf_name = str(args.get("name", ""))
            bridge.read_workflow(wf_name)  # validates it exists and parses
            return _proposal(ctx, "open_workflow", f"Open {wf_name}", name=wf_name, risk="replace")

        if name == "save_workflow":
            import re as _re

            base = _re.sub(r"[^A-Za-z0-9 ._()-]", "", str(args.get("name", ""))).strip()[:80]
            if not base:
                return _json({"ok": False, "errors": ["give the workflow a name"]})
            filename = base if base.lower().endswith(".json") else base + ".json"
            exists = (bridge.workflows_dir() / filename).is_file()
            return _proposal(ctx, "save_workflow",
                             f"{'Overwrite' if exists else 'Save as'} {filename}",
                             name=filename, overwrite=exists, risk="replace" if exists else "edit")

        if name == "queue_run":
            count = max(1, min(int(args.get("count") or 1), 8))
            problems = graph_ops.completeness(graph_ops.VirtualGraph(ctx.snapshot))
            if problems and not ctx.draft_ops:
                return _json({"ok": False, "errors": problems,
                              "fix": "the graph would fail; fix it with the edit tools first"})
            return _proposal(ctx, "queue", f"Run the workflow{f' {count}x' if count > 1 else ''}",
                             count=count, risk="queue")

        return f"Unknown tool {name!r}."
    except Exception as exc:  # noqa: BLE001  (a tool error is data for the model, not a crash)
        return _json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _node_ref(value: Any) -> str:
    """Accept 3, "3", "#3" and refs alike."""
    return str(value).strip().lstrip("#") if value is not None else ""


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    return {}


def _edit_ops(name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    if name == "set_widgets":
        return [{"op": "set_widget", "node": _node_ref(args.get("node")), "name": k, "value": v}
                for k, v in _as_dict(args.get("values")).items()]
    if name == "add_node":
        return [{"op": "add_node", "type": args.get("type"), "ref": str(args.get("ref") or ""),
                 "widgets": _as_dict(args.get("widgets"))}]
    if name == "connect":
        return [{"op": "connect",
                 "from": {"node": _node_ref(args.get("from_node")), "output": args.get("from_output", 0)},
                 "to": {"node": _node_ref(args.get("to_node")), "input": args.get("to_input")}}]
    if name == "disconnect":
        return [{"op": "disconnect", "node": _node_ref(args.get("node")), "input": args.get("input")}]
    if name == "remove_node":
        return [{"op": "remove_node", "node": _node_ref(args.get("node"))}]
    if name == "bypass_node":
        return [{"op": "bypass", "node": _node_ref(args.get("node")), "on": bool(args.get("on", True))}]
    return []


def finalize_draft(ctx: ToolContext, summary: str) -> dict[str, Any] | None:
    """Turn the accumulated edits of this turn into one validated card."""
    if not ctx.draft_ops:
        return None
    result = graph_ops.plan(ctx.draft_ops, ctx.snapshot)
    ctx.draft_ops = []
    if not result["ok"]:
        return None
    if not result["ops"]:
        return None  # every drafted edit was a no-op
    card = {"id": f"p{len(ctx.proposals) + 1}", "kind": "graph_edit",
            "summary": graph_ops.describe(result["diff"], summary), "ops": result["ops"], "diff": result["diff"],
            "warnings": result["warnings"], "risk": result["risk"]}
    ctx.proposals.append(card)
    return card


def _virtual_snapshot(graph: "graph_ops.VirtualGraph") -> dict[str, Any]:
    """Render a VirtualGraph back into snapshot form so it can be summarised."""
    nodes = []
    for nid, node in graph.nodes.items():
        label = nid if nid in graph.preexisting else f"{nid} (new)"
        nodes.append({"id": label, "type": node["type"], "title": node.get("title"),
                      "mode": node.get("mode", 0), "widgets": node.get("widgets") or {},
                      "inputs": [{"name": k, "from": list(v) if v else None}
                                 for k, v in node["inputs"].items()]})
    return {"nodes": nodes}
