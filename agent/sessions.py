"""Chat sessions on disk, one JSON file each, atomic writes."""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from .config import USER_DATA, atomic_write_json

SAFE = re.compile(r"\A[a-z0-9_-]{4,40}\Z")


def _dir():
    return USER_DATA / "chats"


def new_id() -> str:
    return "c_" + uuid.uuid4().hex[:12]


def load(chat_id: str) -> dict[str, Any]:
    if not SAFE.match(chat_id or ""):
        raise ValueError("bad chat id")
    path = _dir() / f"{chat_id}.json"
    if not path.is_file():
        return {"id": chat_id, "title": "", "created": int(time.time()), "turns": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save(chat: dict[str, Any]) -> None:
    if not SAFE.match(chat.get("id", "")):
        raise ValueError("bad chat id")
    chat["updated"] = int(time.time())
    atomic_write_json(_dir() / f"{chat['id']}.json", chat)


def listing(limit: int = 40) -> list[dict[str, Any]]:
    d = _dir()
    if not d.is_dir():
        return []
    rows = []
    for path in d.glob("c_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rows.append({"id": data.get("id"), "title": data.get("title") or "New chat",
                     "updated": data.get("updated", 0), "turns": len(data.get("turns", []))})
    return sorted(rows, key=lambda r: r["updated"], reverse=True)[:limit]


def delete(chat_id: str) -> bool:
    if not SAFE.match(chat_id or ""):
        return False
    path = _dir() / f"{chat_id}.json"
    if path.is_file():
        path.unlink()
        return True
    return False
