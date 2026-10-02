"""Which proposals may apply without a click. Decided here, in code.

The model never chooses this: a proposal's risk is computed by tools.py from
what it actually does, and nothing the model, a remembered note or a skill
says can change the table below.

  ask  mode: every proposal waits for the user.
  auto mode: graph edits and queueing apply immediately (with an undo
             checkpoint); removing nodes that were already on the canvas and
             replacing the whole canvas ALWAYS wait for the user.
"""

from __future__ import annotations

ALWAYS_CONFIRM = {"destructive", "replace"}
AUTO_OK = {"edit", "queue", "none"}


def tools_allowed(mode: str) -> str:
    """inspect = LOOK tools only (the agent cannot even propose a change)."""
    return "look" if mode == "inspect" else "all"


def auto_apply(risk: str, mode: str) -> bool:
    if mode != "auto":
        return False
    if risk in ALWAYS_CONFIRM:
        return False
    return risk in AUTO_OK


def explain(risk: str, mode: str) -> str:
    if auto_apply(risk, mode):
        return "applied automatically (Auto mode); Undo restores the canvas"
    if risk in ALWAYS_CONFIRM:
        return "always needs your click: it removes or replaces work already on the canvas"
    return "waiting for your click (Ask mode)"
