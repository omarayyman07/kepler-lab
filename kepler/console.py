"""Lab Console: a local web page that follows a session folder live.

    python -m kepler console sessions/<name> [--port 8765]

Everything shown is read from files the session already writes: NOTEBOOK.md (the
scientist's reasoning and findings), trialNN.json (technician verdict, laws with
ΔBIC, g), trialNN.overlay.mp4, and console.json (the live step: your turn,
recording, copying, measuring, thinking). The console never changes the session.
Standard library only; it binds to 127.0.0.1.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import status
from .report import _fit_line, parse_notebook

PAGE = Path(__file__).with_name("console.html")


def _num(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _laws(raw: dict[str, list[str]]) -> dict[str, list[dict]]:
    """Fit summary strings -> rows with a verdict: leading (ΔBIC ≤ 2), alive, ruled out (ΔBIC > 10)."""
    out = {}
    for family, lines in raw.items():
        rows = []
        for s in lines:
            f = _fit_line(s)
            d = _num(f.delta_bic)
            verdict = "failed" if d is None else "leading" if d <= 2 else "ruled out" if d > 10 else "alive"
            rows.append({"name": f.name, "dbic": d, "rms": f.rms, "params": f.params, "verdict": verdict})
        out[family] = rows
    return out


def build_state(session: Path) -> dict:
    nb_path = session / "NOTEBOOK.md"
    nb = parse_notebook(nb_path.read_text(encoding="utf-8")) if nb_path.exists() else None
    by_n = {t.n: t for t in nb.trials} if nb else {}
    trials = []
    for p in sorted(session.glob("trial[0-9][0-9].json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):   # being written right now
            continue
        n = int(r.get("trial", p.stem[5:]))
        t = by_n.get(n)
        overlay = Path(r["overlay"]).name if r.get("overlay") else ""
        plot = Path(r["plot"]).name if r.get("plot") else ""
        trials.append({
            "n": n,
            "heading": t.heading if t else r.get("setting", ""),
            "why": t.why if t else "",
            "usable": r.get("usable"),
            "qc_status": r.get("qc_status", ""),
            "qc_issues": r.get("qc_issues", []),
            "qc_notes": r.get("qc_notes", ""),
            "tracker": r.get("tracker", ""),
            "summary": r.get("summary", ""),
            "evidence_failures": r.get("evidence_failures", []),
            "warnings": r.get("warnings", []),
            "laws": _laws(r.get("laws", {})),
            "headline": r.get("headline", {}),
            "overlay": overlay if overlay and (session / overlay).exists() else "",
            "plot": plot if plot and (session / plot).exists() else "",
            "analyses": [a.purpose for a in t.analyses] if t else [],
        })
    # trials planned in the notebook but not measured yet (or refused)
    done = {t["n"] for t in trials}
    planned = [{"n": t.n, "heading": t.heading, "why": t.why} for t in (nb.trials if nb else []) if t.n not in done]
    live = None
    lp = session / "console.json"
    if lp.exists():
        try:
            live = json.loads(lp.read_text(encoding="utf-8"))
            live["age_s"] = time.time() - live.get("updated", time.time())
        except (OSError, json.JSONDecodeError):
            live = None
    return {
        "session": session.name,
        "title": (nb.title.replace("Kepler lab notebook: ", "") if nb and nb.title else "Measurement"),
        "setting": nb.setting if nb else "",
        "provenance": nb.provenance if nb else ("real" if trials else "unknown"),
        "trials": trials,
        "planned": planned,
        "findings": [{"confidence": f.confidence, "claim": f.claim, "evidence": f.evidence}
                     for f in (nb.findings if nb else [])],
        "conclusion": nb.conclusion if nb else "",
        "live": live,
        "now": time.time(),
    }


class Handler(BaseHTTPRequestHandler):
    session: Path

    def log_message(self, *_):   # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):   # DNS-rebinding guard
            return self._send(403, b"local only", "text/plain")
        if path in ("/", "/index.html"):
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/state":
            body = json.dumps(build_state(self.session), default=str).encode()
            return self._send(200, body, "application/json")
        if path.startswith("/files/"):
            return self._file(path[len("/files/"):])
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        """'Ready – release': same as Enter in the session terminal. Accepted only from this
        machine, only in the YOUR TURN stage, and only with that prompt's token."""
        if urlparse(self.path).path != "/api/go":
            return self._send(404, b"not found", "text/plain")
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost") or self.client_address[0] != "127.0.0.1":
            return self._send(403, b"local only", "text/plain")
        try:
            body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 4096)) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._send(400, b"bad request", "text/plain")
        live = {}
        try:
            live = json.loads((self.session / "console.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        if live.get("stage") != "your_turn":
            return self._send(409, b"not your turn", "text/plain")
        if not live.get("go_token") or body.get("token") != live["go_token"]:
            return self._send(403, b"stale or wrong prompt", "text/plain")
        if (self.session / status.GO_FILE).exists():
            return self._send(409, b"already sent", "text/plain")
        tmp = self.session / (status.GO_FILE + ".tmp")
        tmp.write_text(json.dumps({"token": body["token"], "at": time.time()}), encoding="utf-8")
        os.replace(tmp, self.session / status.GO_FILE)
        self._send(200, b'{"ok": true}', "application/json")

    def _file(self, name: str):
        # only plain file names inside the session folder
        if not re.fullmatch(r"[\w.\-]+", name):
            return self._send(400, b"bad name", "text/plain")
        p = (self.session / name).resolve()
        if p.parent != self.session.resolve() or not p.is_file():
            return self._send(404, b"not found", "text/plain")
        size = p.stat().st_size
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip()) if rng else None
        if m and (m[1] or m[2]):   # browsers seek in video with byte ranges
            start = int(m[1]) if m[1] else max(size - int(m[2]), 0)
            end = min(int(m[2]), size - 1) if m[1] and m[2] else size - 1
            end = min(end, start + 8 * 2**20 - 1)   # serve long videos in chunks; the browser asks for more
            if start >= size or start > end:
                return self._send(416, b"", "text/plain", {"Content-Range": f"bytes */{size}"})
            with p.open("rb") as f:
                f.seek(start)
                body = f.read(end - start + 1)
            return self._send(206, body, ctype, {"Content-Range": f"bytes {start}-{end}/{size}",
                                                 "Accept-Ranges": "bytes"})
        self._send(200, p.read_bytes(), ctype, {"Accept-Ranges": "bytes"})


def serve(session: str | Path, port: int = 8765, open_browser: bool = True) -> None:
    session = Path(session).resolve()
    session.mkdir(parents=True, exist_ok=True)   # the console may start before the session does
    handler = type("SessionHandler", (Handler,), {"session": session})
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Lab Console for {session}\n  open {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
