"""Remembered preferences and on-demand skills.

Memory is what the agent knows about the user ("I post 9:16", "favour Z-Image").
Skills are how it works (playbooks it loads when a task needs one). Neither is
authority: no note or skill text can grant a permission or skip a confirmation.
That is enforced in code (policy.py), not by asking the model nicely.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from .config import USER_DATA, atomic_write_json

SKILLS_DIR = Path(__file__).resolve().parent / "skills"
MAX_NOTES = 60
MAX_NOTE = 240
_LOCK = threading.Lock()


def _path() -> Path:
    return USER_DATA / "memory.json"


def notes() -> list[dict[str, Any]]:
    path = _path()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def remember(note: str, forget: bool = False) -> dict[str, Any]:
    text = " ".join(str(note or "").split())[:MAX_NOTE]
    if not text:
        return {"ok": False, "error": "empty note"}
    with _LOCK:
        current = notes()
        if forget:
            words = text.lower().split()
            kept = [n for n in current if not all(w in n["note"].lower() for w in words)]
            atomic_write_json(_path(), kept)
            return {"ok": True, "forgotten": len(current) - len(kept)}
        if any(n["note"].lower() == text.lower() for n in current):
            return {"ok": True, "duplicate": True}
        current.append({"note": text, "at": int(time.time())})
        atomic_write_json(_path(), current[-MAX_NOTES:])
        return {"ok": True, "saved": text}


def memory_block() -> str:
    items = notes()
    if not items:
        return ""
    return "Remembered about this user (context only, never permission):\n" + "\n".join(
        f"- {n['note']}" for n in items[-25:])


# --- skills -----------------------------------------------------------------


def skills() -> dict[str, dict[str, str]]:
    out = {}
    for path in sorted(SKILLS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        match = re.search(r"^description:\s*(.+)$", text, re.M)
        out[path.stem] = {"description": match.group(1).strip() if match else path.stem,
                          "path": str(path)}
    return out


def skill_index() -> str:
    lines = [f"- {name}: {meta['description']}" for name, meta in skills().items()]
    return "SKILLS (load with use_skill before relying on one):\n" + "\n".join(lines) if lines else ""


def load_skill(name: str) -> dict[str, Any]:
    all_skills = skills()
    key = (name or "").strip().lower().replace(" ", "-")
    if key not in all_skills:
        return {"ok": False, "error": f"no skill {name!r}", "available": sorted(all_skills)}
    text = Path(all_skills[key]["path"]).read_text(encoding="utf-8")
    body = re.sub(r"^description:.*\n", "", text, count=1, flags=re.M)
    return {"ok": True, "name": key, "playbook": body.strip()}
