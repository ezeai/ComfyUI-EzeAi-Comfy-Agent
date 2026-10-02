"""Graph edits as a small validated language.

The ComfyUI graph lives in the browser, so the agent never edits it directly.
It emits operations; this module checks every one against the live node
schemas and the current graph snapshot, and returns either a clean plan the
frontend can apply (with an undo checkpoint) or precise errors the model can
correct. A model cannot invent a node type, wire a MODEL into a LATENT socket,
or set steps to "lots".

Operation forms (node = an existing numeric id or a ref named by add_node):

  {"op": "add_node", "ref": "ks", "type": "KSampler", "widgets": {"steps": 20}}
  {"op": "connect", "from": {"node": "ckpt", "output": "MODEL"},
                    "to":   {"node": "ks",   "input": "model"}}
  {"op": "set_widget", "node": 12, "name": "steps", "value": 30}
  {"op": "disconnect", "node": 12, "input": "positive"}
  {"op": "remove_node", "node": 7}
  {"op": "set_title", "node": 12, "title": "Hero sampler"}
  {"op": "bypass", "node": 12, "on": true}
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from . import comfy_bridge as bridge

OPS = ("add_node", "connect", "set_widget", "disconnect", "remove_node", "set_title", "bypass")
MAX_OPS = 60


class VirtualGraph:
    """The graph as it would be after the plan, for validation only."""

    def __init__(self, snapshot: dict[str, Any] | None):
        self.nodes: dict[str, dict[str, Any]] = {}
        self.preexisting: set[str] = set()
        for node in (snapshot or {}).get("nodes", []):
            nid = str(node.get("id"))
            inputs = {}
            for inp in node.get("inputs") or []:
                src = inp.get("from")
                inputs[inp["name"]] = (str(src[0]), int(src[1])) if src else None
            self.nodes[nid] = {"type": node.get("type"), "title": node.get("title") or node.get("type"),
                               "widgets": dict(node.get("widgets") or {}), "inputs": inputs,
                               "mode": node.get("mode", 0)}
            self.preexisting.add(nid)


def _validate_value(spec: dict[str, Any], value: Any) -> tuple[Any, str | None]:
    """Coerce a widget value to its schema type, or explain why not."""
    kind = spec["type"]
    if kind == "COMBO":
        options = spec.get("options") or []
        if str(value) in options:
            return str(value), None
        # forgiving match: case-insensitive, or a unique substring (model files)
        lowered = [o for o in options if o.lower() == str(value).lower()]
        if len(lowered) == 1:
            return lowered[0], None
        partial = [o for o in options if str(value).lower() in o.lower()]
        if len(partial) == 1:
            return partial[0], None
        hint = ", ".join(partial[:6] or options[:8])
        return None, (f"{value!r} is not an option for {spec['name']}"
                      f"{' (ambiguous: ' + hint + ')' if len(partial) > 1 else '; options include ' + hint}")
    if kind == "INT":
        try:
            num = int(float(value))
        except (TypeError, ValueError):
            return None, f"{spec['name']} needs a whole number, got {value!r}"
    elif kind == "FLOAT":
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None, f"{spec['name']} needs a number, got {value!r}"
    elif kind == "BOOLEAN":
        if isinstance(value, bool):
            return value, None
        if str(value).lower() in ("true", "1", "yes", "on"):
            return True, None
        if str(value).lower() in ("false", "0", "no", "off"):
            return False, None
        return None, f"{spec['name']} needs true or false, got {value!r}"
    else:
        return str(value), None
    lo, hi = spec.get("min"), spec.get("max")
    if lo is not None and num < lo:
        return None, f"{spec['name']}={num} is below its minimum {lo}"
    if hi is not None and num > hi:
        return None, f"{spec['name']}={num} is above its maximum {hi}"
    return num, None


def _types_compatible(out_type: str, in_type: str) -> bool:
    if out_type == "*" or in_type == "*":
        return True
    if out_type == in_type:
        return True
    accepted = {t.strip() for t in str(in_type).split(",")}
    return out_type in accepted


def plan(ops: list[dict[str, Any]], snapshot: dict[str, Any] | None,
         *, allow_api_nodes: bool = False) -> dict[str, Any]:
    """Validate a batch. Returns {ok, ops, diff, errors, warnings, risk}."""
    if not isinstance(ops, list) or not ops:
        return {"ok": False, "errors": ["no operations given"], "ops": [], "diff": [], "warnings": []}
    if len(ops) > MAX_OPS:
        return {"ok": False, "errors": [f"too many operations ({len(ops)} > {MAX_OPS}); split the work"],
                "ops": [], "diff": [], "warnings": []}

    graph = VirtualGraph(snapshot)
    refs: dict[str, str] = {}          # add_node ref -> virtual id
    errors: list[str] = []
    out_ops: list[dict[str, Any]] = []
    diff: list[str] = []
    removed_existing: list[str] = []
    counter = 0

    def node_id(value: Any, where: str) -> str | None:
        key = str(value)
        if key in refs:
            return refs[key]
        if key in graph.nodes:
            return key
        errors.append(f"{where}: no node {value!r} in the graph (use an existing id or a ref from add_node)")
        return None

    def label(nid: str) -> str:
        node = graph.nodes.get(nid, {})
        prefix = f"#{nid}" if nid in graph.preexisting else f"new:{nid}"
        return f"{prefix} {node.get('title') or node.get('type')}"

    for i, raw in enumerate(ops):
        where = f"op {i + 1}"
        if not isinstance(raw, dict) or raw.get("op") not in OPS:
            errors.append(f"{where}: unknown operation {raw!r}; use one of {', '.join(OPS)}")
            continue
        op = raw["op"]

        if op == "add_node":
            ntype = str(raw.get("type", ""))
            schema = bridge.schema(ntype)
            if schema is None:
                import difflib

                close = difflib.get_close_matches(ntype, list(bridge.object_info()), n=3, cutoff=0.6)                     or [c["type"] for c in bridge.search_nodes(ntype, limit=3)]
                hint = f" Did you mean: {', '.join(close)}?" if close else ""
                errors.append(f"{where}: node type {ntype!r} is not installed.{hint}")
                continue
            if schema["api_node"] and not allow_api_nodes:
                errors.append(f"{where}: {ntype} is a paid cloud API node; local-first agents may not add it")
                continue
            counter += 1
            ref = str(raw.get("ref") or f"n{counter}")
            vid = f"v{counter}"
            if ref in refs:
                errors.append(f"{where}: ref {ref!r} is already used in this plan")
                continue
            refs[ref] = vid
            widgets_in = raw.get("widgets") or {}
            specs = {s["name"]: s for s in schema["inputs"] if s["widget"]}
            widgets_out = {}
            for name, value in widgets_in.items():
                if name not in specs:
                    errors.append(f"{where}: {ntype} has no widget {name!r} "
                                  f"(widgets: {', '.join(specs) or 'none'})")
                    continue
                coerced, problem = _validate_value(specs[name], value)
                if problem:
                    errors.append(f"{where}: {problem}")
                else:
                    widgets_out[name] = coerced
            graph.nodes[vid] = {"type": ntype, "title": raw.get("title") or schema["display"],
                                "widgets": widgets_out,
                                "inputs": {s["name"]: None for s in schema["inputs"] if not s["widget"]},
                                "mode": 0}
            out_ops.append({"op": "add_node", "ref": ref, "vid": vid, "type": ntype,
                            "widgets": widgets_out, "title": raw.get("title")})
            diff.append(f"+ {schema['display']} ({ref})")

        elif op == "connect":
            src, dst = raw.get("from") or {}, raw.get("to") or {}
            sid = node_id(src.get("node"), where)
            did = node_id(dst.get("node"), where)
            if sid is None or did is None:
                continue
            s_schema = bridge.schema(graph.nodes[sid]["type"])
            d_schema = bridge.schema(graph.nodes[did]["type"])
            if s_schema is None or d_schema is None:
                errors.append(f"{where}: cannot read the schema of a node in this connection")
                continue
            want = src.get("output", 0)
            output = None
            for o in s_schema["outputs"]:
                if str(want) in (str(o["index"]), o["name"], o["type"]):
                    output = o
                    break
            if output is None:
                errors.append(f"{where}: {label(sid)} has no output {want!r} "
                              f"(outputs: {', '.join(o['name'] for o in s_schema['outputs'])})")
                continue
            in_name = str(dst.get("input", ""))
            spec = next((s for s in d_schema["inputs"] if s["name"] == in_name), None)
            if spec is None:
                links = [s["name"] for s in d_schema["inputs"] if not s["widget"]]
                errors.append(f"{where}: {label(did)} has no input {in_name!r} (inputs: {', '.join(links)})")
                continue
            if not _types_compatible(output["type"], spec["type"]):
                errors.append(f"{where}: cannot connect {output['type']} into "
                              f"{label(did)}.{in_name} which expects {spec['type']}")
                continue
            graph.nodes[did]["inputs"][in_name] = (sid, output["index"])
            out_ops.append({"op": "connect", "from": {"node": sid, "output": output["index"]},
                            "to": {"node": did, "input": in_name},
                            "converts_widget": bool(spec["widget"])})
            diff.append(f"→ {label(sid)}.{output['name']} ⟶ {label(did)}.{in_name}")

        elif op == "set_widget":
            nid = node_id(raw.get("node"), where)
            if nid is None:
                continue
            schema = bridge.schema(graph.nodes[nid]["type"])
            spec = next((s for s in (schema or {}).get("inputs", [])
                         if s["name"] == raw.get("name") and s["widget"]), None)
            if spec is None:
                names = [s["name"] for s in (schema or {}).get("inputs", []) if s["widget"]]
                errors.append(f"{where}: {label(nid)} has no widget {raw.get('name')!r} "
                              f"(widgets: {', '.join(names) or 'none'})")
                continue
            coerced, problem = _validate_value(spec, raw.get("value"))
            if problem:
                errors.append(f"{where}: {problem}")
                continue
            before = graph.nodes[nid]["widgets"].get(spec["name"])
            if _same(before, coerced):
                # Measured: a small model re-"set" five values to what they
                # already were. A no-op is not a change; keep it out of the card.
                continue
            graph.nodes[nid]["widgets"][spec["name"]] = coerced
            out_ops.append({"op": "set_widget", "node": nid, "name": spec["name"], "value": coerced})
            diff.append(f"~ {label(nid)}.{spec['name']}: {_short(before)} → {_short(coerced)}")

        elif op == "disconnect":
            nid = node_id(raw.get("node"), where)
            if nid is None:
                continue
            name = str(raw.get("input", ""))
            if name not in graph.nodes[nid]["inputs"]:
                errors.append(f"{where}: {label(nid)} has no linked input {name!r}")
                continue
            graph.nodes[nid]["inputs"][name] = None
            out_ops.append({"op": "disconnect", "node": nid, "input": name})
            diff.append(f"✕ {label(nid)}.{name} unlinked")

        elif op == "remove_node":
            nid = node_id(raw.get("node"), where)
            if nid is None:
                continue
            if nid in graph.preexisting:
                removed_existing.append(nid)
            gone = graph.nodes.pop(nid)
            for other in graph.nodes.values():
                for key, src in list(other["inputs"].items()):
                    if src and src[0] == nid:
                        other["inputs"][key] = None
            out_ops.append({"op": "remove_node", "node": nid})
            diff.append(f"− #{nid} {gone.get('title') or gone.get('type')}")

        elif op == "set_title":
            nid = node_id(raw.get("node"), where)
            if nid is None:
                continue
            title = str(raw.get("title", ""))[:120]
            graph.nodes[nid]["title"] = title
            out_ops.append({"op": "set_title", "node": nid, "title": title})
            diff.append(f"~ {label(nid)} renamed")

        elif op == "bypass":
            nid = node_id(raw.get("node"), where)
            if nid is None:
                continue
            on = bool(raw.get("on", True))
            graph.nodes[nid]["mode"] = 4 if on else 0
            out_ops.append({"op": "bypass", "node": nid, "on": on})
            diff.append(f"{'⊘' if on else '◉'} {label(nid)} {'bypassed' if on else 'enabled'}")

    warnings = completeness(graph)
    risk = "destructive" if removed_existing else ("edit" if out_ops else "none")
    return {"ok": not errors, "ops": out_ops if not errors else [], "diff": diff,
            "errors": errors, "warnings": warnings, "risk": risk,
            "removed_existing": removed_existing, "refs": refs, "graph": graph}


def completeness(graph: VirtualGraph) -> list[str]:
    """Problems that would make the graph fail when queued."""
    warnings = []
    has_output = False
    for nid, node in graph.nodes.items():
        if node.get("mode") == 4:
            continue
        schema = bridge.schema(node["type"])
        if schema is None:
            warnings.append(f"#{nid} {node['type']}: node type is not installed")
            continue
        if schema["output_node"]:
            has_output = True
        for spec in schema["inputs"]:
            if spec["required"] and not spec["widget"] and not node["inputs"].get(spec["name"]):
                warnings.append(f"{node.get('title') or node['type']} ({nid}): required input "
                                f"'{spec['name']}' ({spec['type']}) is not connected")
    if graph.nodes and not has_output:
        warnings.append("no output node (e.g. SaveImage / PreviewImage): queueing would do nothing")
    warnings.extend(_bypassed_passthroughs(graph))
    return warnings


def _bypassed_passthroughs(graph: VirtualGraph) -> list[str]:
    """A new pass-through node whose output is unused while its source still
    feeds the old consumers directly.

    Measured: a 4B model added a LoRA, routed its MODEL output to the sampler,
    but left both text encoders reading the checkpoint CLIP. The graph was
    "complete" (no missing inputs), so nothing flagged it, yet the LoRA CLIP
    strength silently did nothing. This catches exactly that shape.
    """
    out = []
    used: set[tuple[str, int]] = set()
    for node in graph.nodes.values():
        for src in node["inputs"].values():
            if src:
                used.add((src[0], int(src[1])))
    for nid, node in graph.nodes.items():
        if nid in graph.preexisting:
            continue
        schema = bridge.schema(node["type"])
        if schema is None:
            continue
        for spec in schema["inputs"]:
            if spec["widget"]:
                continue
            src = node["inputs"].get(spec["name"])
            if not src:
                continue
            same = next((o for o in schema["outputs"] if o["type"] == spec["type"]), None)
            if same is None or (nid, same["index"]) in used:
                continue
            # the new node's output of this type is unused, yet its source
            # output still feeds other nodes directly
            others = [n for n, other in graph.nodes.items() if n != nid and any(
                s and s[0] == src[0] and int(s[1]) == int(src[1]) for s in other["inputs"].values())]
            if others:
                out.append(f"{node.get('title') or node['type']} ({nid}): its {spec['type']} output is "
                           f"unused while {len(others)} node(s) still read #{src[0]} directly, so it has "
                           f"no effect on them. Use insert_after to rewire them all.")
    return out


def summarize(snapshot: dict[str, Any] | None, limit: int = 80) -> str:
    """The current graph as compact text for the model."""
    nodes = (snapshot or {}).get("nodes") or []
    if not nodes:
        return "The canvas is empty."
    lines = [f"{len(nodes)} nodes on the canvas"
             + (f" (showing {limit})" if len(nodes) > limit else "") + ":"]
    for node in nodes[:limit]:
        title = node.get("title") or node.get("type")
        mode = {2: " [muted]", 4: " [bypassed]"}.get(node.get("mode", 0), "")
        widgets = ", ".join(f"{k}={_short(v)}" for k, v in (node.get("widgets") or {}).items())
        links = ", ".join(f"{i['name']}<-#{i['from'][0]}.{i['from'][1]}"
                          for i in node.get("inputs") or [] if i.get("from"))
        lines.append(f"#{node.get('id')} {node.get('type')}"
                     + (f" \"{title}\"" if title != node.get("type") else "") + mode
                     + (f" | {widgets}" if widgets else "") + (f" | in: {links}" if links else ""))
    problems = completeness(VirtualGraph(snapshot))
    missing = [p for p in problems if p.endswith("node type is not installed")]
    if missing:
        # Listed first and in full: a 4B model must not try to rewire a node that cannot run.
        lines.insert(1, "MISSING NODES (cannot run, do not rewire): "
                     + "; ".join(p.split(": ")[0] for p in missing)
                     + ". Replace the workflow with a create_* tool, or tell the user which node pack to install.")
    others = [p for p in problems if p not in missing]
    if others:
        lines.append("Problems: " + "; ".join(others[:8]))
    return "\n".join(lines)


def _same(a: Any, b: Any) -> bool:
    if a is None:
        return False
    try:
        if isinstance(b, (int, float)) and not isinstance(b, bool):
            return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return False
    return str(a) == str(b)


def describe(diff: list[str], fallback: str = "Edit the workflow") -> str:
    """A card title written from what the change does, not from model prose."""
    if not diff:
        return fallback
    adds = [d for d in diff if d.startswith("+")]
    sets = [d for d in diff if d.startswith("~")]
    links = [d for d in diff if d.startswith("→")]
    removes = [d for d in diff if d.startswith(("−", "✕"))]
    parts = []
    if adds:
        names = [d[2:].split(" (")[0] for d in adds]
        parts.append("Add " + (names[0] if len(names) == 1 else f"{len(names)} nodes"))
    if sets:
        fields = sorted({d.split(":")[0].split(".")[-1] for d in sets})
        parts.append("Change " + ", ".join(fields[:3]) + ("…" if len(fields) > 3 else ""))
    if links and not adds:
        parts.append(f"Rewire {len(links)} link{'s' if len(links) > 1 else ''}")
    if removes:
        parts.append(f"Remove {len(removes)}")
    return " · ".join(parts) or fallback


def _short(value: Any, n: int = 48) -> str:
    text = str(value)
    return text if len(text) <= n else text[: n - 1] + "…"


def consumers(snapshot: dict[str, Any] | None, draft: list[dict[str, Any]],
              source: str, slot: int) -> list[tuple[str, str]]:
    """Every (node, input) currently fed by source.slot, pending edits included."""
    result = plan(draft, snapshot) if draft else {"graph": VirtualGraph(snapshot), "ok": True}
    graph = result.get("graph") or VirtualGraph(snapshot)
    out = []
    for nid, node in graph.nodes.items():
        for name, src in node["inputs"].items():
            if src and src[0] == str(source) and int(src[1]) == int(slot):
                out.append((nid, name))
    return out


def insert_after_ops(snapshot: dict[str, Any] | None, draft: list[dict[str, Any]],
                     source: str, node_type: str, ref: str,
                     widgets: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Insert a pass-through node after `source`, rewiring all its consumers.

    For each input of the new node whose type matches an output of the
    source (LoraLoader: MODEL and CLIP; ModelSamplingAuraFlow: MODEL; an image
    filter: IMAGE), the source output is routed into the new node, and every
    node that used to read that source output now reads the new node's output
    of the same type instead. This is the "add a LoRA between the checkpoint
    and the sampler" pattern, done deterministically.
    """
    problems: list[str] = []
    new_schema = bridge.schema(node_type)
    if new_schema is None:
        return [], [f"node type {node_type!r} is not installed"]
    base = plan(draft, snapshot) if draft else {"graph": VirtualGraph(snapshot), "refs": {}}
    graph = base.get("graph") or VirtualGraph(snapshot)
    refs = base.get("refs") or {}
    src_id = refs.get(str(source), str(source))
    if src_id not in graph.nodes:
        return [], [f"no node {source!r} to insert after"]
    src_schema = bridge.schema(graph.nodes[src_id]["type"])
    if src_schema is None:
        return [], ["cannot read the source node schema"]

    ops: list[dict[str, Any]] = [{"op": "add_node", "type": node_type, "ref": ref, "widgets": widgets}]
    routed = 0
    for spec in new_schema["inputs"]:
        if spec["widget"]:
            continue
        src_out = next((o for o in src_schema["outputs"] if o["type"] == spec["type"]), None)
        if src_out is None:
            continue
        new_out = next((o for o in new_schema["outputs"] if o["type"] == spec["type"]), None)
        # rewire consumers first (they read the source output today)
        if new_out is not None:
            for node_id, input_name in consumers(snapshot, draft, src_id, src_out["index"]):
                ops.append({"op": "connect", "from": {"node": ref, "output": new_out["index"]},
                            "to": {"node": node_id, "input": input_name}})
        ops.append({"op": "connect", "from": {"node": src_id, "output": src_out["index"]},
                    "to": {"node": ref, "input": spec["name"]}})
        routed += 1
    if not routed:
        problems.append(f"{node_type} has no input whose type matches an output of node {source}")
    return ops, problems
