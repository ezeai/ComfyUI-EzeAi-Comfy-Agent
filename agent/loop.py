"""The agent loop.

LOOK -> think -> (LOOK more | PROPOSE) -> answer, for at most `max_rounds`.

Context is compiled, not dumped: each request carries the rules, the skill
index, remembered notes, a bounded summary of the canvas, and only the TEXT of
earlier turns - never stale tool output from previous turns. The manifest of
what went in (and what was dropped) is returned for the Inspector.

Small local models loop; two cheap guards stop that: an identical repeated tool
call gets its cached result plus a nudge, and proposals are capped.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable

from . import graph_ops, memory, models, policy, providers, recipes, tools
from .config import load_settings

Emit = Callable[[dict[str, Any]], None]
MAX_PROPOSALS = 4

# Measured: asked to save a workflow, a 4B model replied "I've saved the current
# canvas" without calling any tool. These patterns flag an answer that claims an
# action; the loop then verifies a DO tool actually ran.
import re as _re

_CLAIM = _re.compile(
    r"\b(i'?ve|i have|i)\s+(just\s+)?(saved|added|created|built|changed|updated|set|connected|"
    r"removed|deleted|queued|run|ran|started|applied|made|replaced|inserted|opened|loaded)\b"
    r"|\b(saved|queued|applied) (it|the|your)\b|^\s*done\b",
    _re.I | _re.M)
HISTORY_TURNS = 8

IDENTITY = """You are EzeAi Comfy Agent, a local assistant inside ComfyUI. You run entirely on the
user's own PC. You help them understand, build, fix and run ComfyUI workflows.

HOW YOU WORK
1. LOOK before you act or answer: read_graph for anything about the canvas; search_nodes and
   node_info before using an unfamiliar node; list_models before naming a model file;
   check_last_run when something failed; look_at_output before judging an image.
2. NEW IMAGE from scratch -> call the matching create_* tool (a proven workflow) with a complete,
   specific prompt. Do not wire it by hand.
3. CHANGE the canvas with set_widgets, add_node, connect, disconnect, remove_node, bypass_node.
   To put a node BETWEEN others (a LoRA after the checkpoint, a model patch, an image filter)
   use insert_after - it rewires every connection for you. Use node #ids as read_graph shows
   them. All edits in one turn become ONE card; read_graph shows them as PENDING.
4. RUN with queue_run only when the graph is complete. open_workflow loads a saved workflow.
5. Every change is a card the user sees. Never claim you changed or ran anything.
6. If a tool returns errors, fix the call and try again. Never invent node types, widget names,
   values or filenames - look them up.
7. A node the canvas shows as "not installed" cannot run and cannot be rewired. Do not fiddle with
   it: for a new image use a create_* tool; otherwise tell the user which node pack is missing.
   A request to "create"/"make" something new is a NEW IMAGE: skip search_nodes, write the prompt.
8. Answer in plain text: short, specific, friendly. Under 120 words unless asked for more.
   Say what you looked at and what the card will do.

SPECIFIC BEATS GENERIC: prompts you write name the subject, action, setting, light and framing
concretely; when there are N items, state N and describe each one."""


class Agent:
    def __init__(self, emit: Emit, cancel: threading.Event | None = None):
        self.emit = emit
        self.cancel = cancel or threading.Event()

    # --- context ---------------------------------------------------------

    def _system(self, snapshot: dict | None) -> tuple[str, list[dict[str, Any]]]:
        manifest: list[dict[str, Any]] = []

        def part(name: str, text: str, budget: int) -> str:
            if not text:
                return ""
            kept = text if len(text) <= budget else text[:budget] + "\n…(trimmed)"
            manifest.append({"part": name, "chars": len(kept), "dropped": max(0, len(text) - budget)})
            return kept

        try:
            recipe_text = "WORKFLOW TOOLS available: " + ", ".join(
                f"{v['tool']} ({v['label']})" for v in recipes.available().values())
        except Exception:  # noqa: BLE001
            recipe_text = ""
        blocks = [
            part("identity", IDENTITY, 4000),
            part("skills", memory.skill_index(), 1500),
            part("memory", memory.memory_block(), 1800),
            part("recipes", recipe_text, 800),
            part("canvas", "CANVAS NOW (read_graph gives the same, refreshed):\n"
                 + graph_ops.summarize(snapshot, limit=40), 5000),
        ]
        return "\n\n".join(b for b in blocks if b), manifest

    def _vision_fn(self, agent_model: dict) -> Callable[[str, str], str] | None:
        settings = load_settings()
        try:
            vision = models.resolve(settings.get("vision_model") or None, need=("vision",))
        except Exception:  # noqa: BLE001
            return None
        if vision["ref"] == agent_model["ref"]:
            return None

        def describe(b64: str, question: str) -> str:
            self.emit({"type": "route", "step": "vision", "model": vision["name"]})
            reply = providers.chat(vision, [
                {"role": "system", "content": "You describe images precisely for another assistant. "
                                              "Be concrete and brief; list visible problems."},
                {"role": "user", "content": question, "images": [b64]}],
                cancel=self.cancel, options={"keep_alive": "60s", "think": False})
            return reply["content"].strip()
        return describe

    def _wrap_up(self, model, messages, settings, why: str) -> str:
        """One last turn with no tools: say what was learned and what is needed next."""
        self.emit({"type": "turn_break"})
        messages.append({"role": "user", "content":
                         f"{why} Stop calling tools. In plain text, in under 80 words: say what you found "
                         "or prepared (mention the pending card if there is one) and ask the user ONE "
                         "concrete question or give the one next step."})
        reply = providers.chat(model, messages, [], cancel=self.cancel,
                               on_delta=lambda kind, text: self.emit({"type": "delta", "kind": kind, "text": text}),
                               options={"think": False, "context": settings["context_tokens"]})
        return reply["content"].strip()

    # --- the loop --------------------------------------------------------

    def run(self, chat: dict[str, Any], user_text: str, snapshot: dict | None,
            images: list[str] | None = None, canvas_image: str | None = None) -> dict[str, Any]:
        settings = load_settings()
        started = time.time()
        # Wall-clock limit: the run cancels itself rather than holding the
        # model forever. Cancellation is checked between every streamed chunk.
        timer = threading.Timer(float(settings["run_timeout_s"]), self.cancel.set)
        timer.daemon = True
        timer.start()
        try:
            return self._run(chat, user_text, snapshot, images, settings, started, canvas_image)
        finally:
            timer.cancel()

    def _run(self, chat, user_text, snapshot, images, settings, started, canvas_image=None):
        model = models.resolve(settings.get("agent_model") or None, need=("tools",)) \
            if not settings.get("agent_model") else models.resolve(settings["agent_model"])
        mode = settings["permission_mode"]
        self.emit({"type": "route", "step": "agent", "model": model["name"],
                   "provider": model["provider"], "mode": mode,
                   "caps": model["caps"]})

        system, manifest = self._system(snapshot)
        if mode == "inspect":
            system += ("\n\nMODE: Inspect (read-only). You have only LOOK tools. Explain, diagnose and "
                       "advise; if a change would help, describe it so the user can switch to Ask or "
                       "Autonomous mode.")
        elif mode == "ask":
            system += ("\n\nMODE: Ask. Nothing you propose is applied until the user clicks Apply. "
                       "Say \"I've prepared a card that...\", never \"Done\" or \"I've added\".")
        else:
            system += ("\n\nMODE: Autonomous. Your edits apply immediately (with Undo). When the user "
                       "wants an image or a result, finish the workflow AND call queue_run: the run's "
                       "output is shown back to you automatically so you can review and improve it. "
                       "Removing or replacing existing work still waits for the user.")
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for turn in (chat.get("turns") or [])[-HISTORY_TURNS:]:
            messages.append({"role": "user", "content": turn.get("user", "")})
            reply = turn.get("assistant", "")
            if turn.get("proposals"):
                reply += ("\n(System note: these were only PROPOSED as cards; whether the user applied "
                          "them is unknown - trust the CANVAS NOW section: "
                          + "; ".join(p.get("summary", "") for p in turn["proposals"]) + ")")
            messages.append({"role": "assistant", "content": reply})
        manifest.append({"part": "history", "turns": min(len(chat.get("turns") or []), HISTORY_TURNS)})

        ctx = tools.ToolContext(snapshot=snapshot, agent_sees=bool(model["caps"].get("vision")),
                                canvas_image=canvas_image)
        ctx.vision = self._vision_fn(model)

        user_msg: dict[str, Any] = {"role": "user", "content": user_text}
        if images:
            if model["caps"].get("vision"):
                user_msg["images"] = images
            elif ctx.vision:
                notes = [ctx.vision(b64, "Describe this attached image for the assistant.") for b64 in images]
                user_msg["content"] += "\n\n[Attached image, described by the vision model]\n" + "\n".join(notes)
            else:
                user_msg["content"] += "\n\n[An image was attached but no vision model is installed.]"
        messages.append(user_msg)

        tool_specs = tools.all_tools(mode)
        seen_calls: dict[str, str] = {}
        repeats: dict[str, int] = {}
        stuck = False
        claim_checked = False
        unverified_claim = False
        rounds = 0
        final_text = ""
        usage = {"prompt": 0, "completion": 0}
        trace: list[dict[str, Any]] = []
        max_rounds = int(settings["max_rounds"])

        while rounds < max_rounds:
            rounds += 1
            if self.cancel.is_set():
                raise providers.Cancelled("stopped by the user")
            self.emit({"type": "round", "n": rounds})
            reply = providers.chat(
                model, messages, tool_specs,
                on_delta=lambda kind, text: self.emit({"type": "delta", "kind": kind, "text": text}),
                cancel=self.cancel,
                options={"think": settings["think"], "context": settings["context_tokens"]})
            for k in ("prompt", "completion"):
                usage[k] += int((reply.get("usage") or {}).get(k) or 0)

            calls = reply["tool_calls"]
            if not calls:
                final_text = reply["content"].strip()
                acted = bool(ctx.proposals or ctx.draft_ops)
                if mode != "inspect" and not acted and _CLAIM.search(final_text) and not claim_checked:
                    claim_checked = True
                    self.emit({"type": "route", "step": "claim_check",
                               "detail": "answer claimed an action but no tool ran"})
                    self.emit({"type": "turn_break"})
                    messages.append({"role": "assistant", "content": final_text})
                    messages.append({"role": "user", "content":
                                     "System check: your answer says you did something, but you called no "
                                     "tool, so NOTHING happened. Call the right tool now (for example "
                                     "save_workflow, set_widgets, queue_run). If you cannot, tell the user "
                                     "plainly that it has not been done."})
                    continue
                if mode != "inspect" and not acted and _CLAIM.search(final_text):
                    unverified_claim = True
                break

            messages.append({"role": "assistant", "content": reply["content"], "tool_calls": calls})
            self.emit({"type": "turn_break"})  # content streamed so far was pre-tool narration
            for call in calls:
                if self.cancel.is_set():
                    raise providers.Cancelled("stopped by the user")
                name, args = call["name"], call.get("arguments") or {}
                key = name + json.dumps(args, sort_keys=True, default=str)
                t0 = time.time()
                if name in tools.DO_NAMES and len(ctx.proposals) >= MAX_PROPOSALS:
                    result = "Proposal limit reached for this turn; summarise for the user instead."
                elif key in seen_calls:
                    repeats[key] = repeats.get(key, 1) + 1
                    result = seen_calls[key] + "\n(You already made this exact call. Use this result.)"
                    if repeats[key] >= 3:
                        # the model is stuck re-reading; stop and let it answer
                        stuck = True
                else:
                    before = len(ctx.proposals)
                    result = tools.run(name, args, ctx)
                    seen_calls[key] = result
                    for card in ctx.proposals[before:]:
                        card["auto_apply"] = policy.auto_apply(card.get("risk", ""), mode)
                        card["policy"] = policy.explain(card.get("risk", ""), mode)
                        self.emit({"type": "proposal", "card": card})
                ms = int((time.time() - t0) * 1000)
                kind = "look" if name in tools.LOOK_NAMES else "do"
                self.emit({"type": "tool", "name": name, "kind": kind, "args": args,
                           "result": result[:1200], "ms": ms})
                trace.append({"name": name, "kind": kind, "ms": ms})
                tool_msg = {"role": "tool", "tool_call_id": call.get("id"), "tool_name": name,
                            "content": result}
                messages.append(tool_msg)
            if ctx.attached_images:
                messages.append({"role": "user", "content": "(the requested image)",
                                 "images": list(ctx.attached_images)})
                ctx.attached_images.clear()
            if stuck:
                self.emit({"type": "route", "step": "stuck", "detail": "repeated identical tool calls"})
                final_text = self._wrap_up(model, messages, settings, "You are repeating the same calls.")
                break
        else:
            self.emit({"type": "route", "step": "limit", "detail": f"step limit ({max_rounds}) reached"})
            final_text = self._wrap_up(model, messages, settings, "You have used all your steps.")

        card = tools.finalize_draft(ctx, _first_line(final_text) or "Edit the workflow")
        if card:
            card["auto_apply"] = policy.auto_apply(card["risk"], mode)
            card["policy"] = policy.explain(card["risk"], mode)
            self.emit({"type": "proposal", "card": card})

        return {
            "text": final_text or "(no answer)",
            "proposals": ctx.proposals,
            "model": model["name"],
            "provider": model["provider"],
            "rounds": rounds,
            "usage": usage,
            "elapsed": round(time.time() - started, 1),
            "manifest": manifest,
            "trace": trace,
            # Code-truth, shown by the UI regardless of what the text says.
            "unverified_claim": unverified_claim,
        }


def _first_line(text: str, n: int = 90) -> str:
    line = (text or "").strip().splitlines()[0].strip(" *#") if (text or "").strip() else ""
    return line if len(line) <= n else line[: n - 1] + "…"
