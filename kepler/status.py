"""Live status of a running session, for the Lab Console (`python -m kepler console`).

The lab writes one small JSON file, `<session>/console.json`, at each step: the
person's turn (with the exact instructions), recording (with its end time),
waiting for the clip, measuring, the scientist thinking, done. Writes are atomic
(temp file + rename) so the console never reads half a file. With no path set
(tests, `measure`) every call is a no-op.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

_path: Path | None = None
_state: dict = {}
# kept across stage changes: the session's budget and spend, and the current trial's plan
_SESSION_FIELDS = ("trial", "budget", "usd_spent", "usd_cap", "setting", "rationale")


def set_session(session_dir: str | Path | None) -> None:
    global _path, _state
    _path = Path(session_dir) / "console.json" if session_dir else None
    _state = {}


def write(stage: str, **fields) -> None:
    """Replace the live stage; keeps session-wide fields (trial, spend) unless given."""
    if _path is None:
        return
    keep = {k: _state[k] for k in _SESSION_FIELDS if k in _state}
    _state.clear()
    _state.update(keep, stage=stage, updated=time.time(), **fields)
    tmp = _path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_state, default=str), encoding="utf-8")
    os.replace(tmp, _path)


def current() -> dict:
    return dict(_state)


GO_FILE = "console_go.json"


def go_path() -> Path | None:
    """Where the console drops its 'Ready – release' signal; None outside a live session."""
    return _path.with_name(GO_FILE) if _path is not None else None


def wait_for_go(prompt: str, token: str) -> str:
    """Block until Enter in the terminal OR the console's go file (with this prompt's token).
    Returns "terminal" or "console". The file is deleted when used."""
    import sys

    go = go_path()
    if go is None:
        input(prompt)
        return "terminal"
    print(prompt, end="", flush=True)
    tty = sys.stdin is not None and sys.stdin.isatty()
    while True:
        if go.exists():
            try:
                ok = json.loads(go.read_text(encoding="utf-8")).get("token") == token
            except (OSError, json.JSONDecodeError):
                ok = False
            go.unlink(missing_ok=True)
            if ok:
                print("\n>>> Ready pressed in the Lab Console.")
                return "console"
        if tty and _enter_pressed():
            return "terminal"
        time.sleep(0.1)


def _enter_pressed() -> bool:
    """Non-blocking check for Enter on the terminal (Windows console or POSIX tty)."""
    import sys

    if os.name == "nt":
        import msvcrt
        while msvcrt.kbhit():
            if msvcrt.getwch() in ("\r", "\n"):
                print()
                return True
        return False
    import select
    r, _, _ = select.select([sys.stdin], [], [], 0)
    if r:
        sys.stdin.readline()
        return True
    return False


def update(**fields) -> None:
    """Change fields without changing the stage (e.g. spend after a model call)."""
    if _path is None:
        return
    _state.update(fields, updated=time.time())
    write(_state.get("stage", "idle"), **{k: v for k, v in _state.items()
                                          if k not in ("stage", "updated", *_SESSION_FIELDS)})
