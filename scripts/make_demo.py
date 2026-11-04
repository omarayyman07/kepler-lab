"""Stitch a session into a short demo video: title cards, the scientist's decisions,
overlay clips, plots and findings.

    python scripts/make_demo.py sessions/<name> [--out demo.mp4] [--clip-seconds 6] [--speed 1]

Everything on screen comes from the session folder: text is quoted from NOTEBOOK.md
(the scientist's own words, shortened only by cutting at a sentence end and marking
the cut with "…"), numbers are the notebook's. Every frame carries the provenance
label (SIMULATED / REAL / PROVENANCE NOT RECORDED), so no clip can be lifted out of
context.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kepler.report import PROVENANCE, Notebook, load  # noqa: E402

W, H, FPS = 1280, 720, 30
BG = (18, 18, 20)
FG = (236, 236, 236)
MUTED = (165, 165, 170)
ACCENT = (240, 160, 80)
LABEL_COLOURS = {"simulated": (255, 194, 122), "real": (123, 224, 170), "unknown": (200, 200, 200)}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    from matplotlib import font_manager

    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans",
                                                             weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=font) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def excerpt(text: str, limit: int = 300) -> str:
    """Whole sentences of the scientist's text up to `limit` characters; a cut is marked with …"""
    sentences = re.split(r"(?<=[.;!?])\s+", text.strip())
    out = ""
    for s in sentences:
        if len(out) + len(s) + 1 > limit:
            break
        out = f"{out} {s}".strip()
    if not out:   # one long sentence: cut at a word boundary
        out = text[:limit].rsplit(" ", 1)[0]
    return out + (" …" if len(out) < len(text.strip()) else "")


class Demo:
    def __init__(self, nb: Notebook, out: Path):
        self.nb = nb
        self.label = PROVENANCE[nb.provenance][0]
        self.label_colour = LABEL_COLOURS[nb.provenance]
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
             "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
            stdin=subprocess.PIPE)
        self.frames = 0
        self.f_label = _font(20, bold=True)

    # ---- low level ---------------------------------------------------------

    def _stamp(self, img: Image.Image) -> Image.Image:
        d = ImageDraw.Draw(img)
        tw = d.textlength(self.label, font=self.f_label)
        x0, y0 = W - tw - 36, 18
        d.rounded_rectangle([x0 - 12, y0 - 6, W - 24, y0 + 30], radius=8, fill=(0, 0, 0))
        d.text((x0, y0), self.label, font=self.f_label, fill=self.label_colour)
        return img

    def _emit(self, img: Image.Image, seconds: float, fade: float = 0.3) -> None:
        bgr = cv2.cvtColor(np.asarray(self._stamp(img.convert("RGB"))), cv2.COLOR_RGB2BGR)
        n = max(1, int(round(seconds * FPS)))
        nf = int(fade * FPS)
        for k in range(n):
            a = min(1.0, (k + 1) / nf) if nf else 1.0
            self.proc.stdin.write((bgr * a).astype(np.uint8).tobytes() if a < 1 else bgr.tobytes())
        self.frames += n

    def close(self) -> float:
        self.proc.stdin.close()
        self.proc.wait()
        return self.frames / FPS

    # ---- segments ----------------------------------------------------------

    def card(self, kicker: str, title: str, body: str = "", seconds: float = 4.0, body_note: str = "") -> None:
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        y = 120
        if kicker:
            d.text((100, y), kicker.upper(), font=_font(24, True), fill=ACCENT)
            y += 50
        for ln in _wrap(d, title, _font(46, True), W - 200)[:3]:
            d.text((100, y), ln, font=_font(46, True), fill=FG)
            y += 60
        y += 20
        if body_note:
            d.text((100, y), body_note, font=_font(20), fill=MUTED)
            y += 36
        f = _font(28)
        for ln in _wrap(d, body, f, W - 200)[:9]:
            d.text((100, y), ln, font=f, fill=FG if not body_note else (215, 215, 215))
            y += 40
        self._emit(img, seconds)

    def still(self, path: Path, caption: str, seconds: float = 4.0) -> None:
        img = Image.new("RGB", (W, H), BG)
        pic = Image.open(path).convert("RGB")
        scale = min((W - 80) / pic.width, (H - 150) / pic.height)
        pic = pic.resize((int(pic.width * scale), int(pic.height * scale)), Image.LANCZOS)
        img.paste(pic, ((W - pic.width) // 2, 70))
        d = ImageDraw.Draw(img)
        f = _font(24)
        y = 70 + pic.height + 18
        for ln in _wrap(d, caption, f, W - 160)[:2]:
            d.text(((W - d.textlength(ln, font=f)) / 2, y), ln, font=f, fill=FG)
            y += 32
        self._emit(img, seconds)

    def clip(self, path: Path, seconds: float, speed: float, caption: str) -> None:
        cap = cv2.VideoCapture(str(path))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_out = int(seconds * FPS)
        want = [int(round(k / FPS * speed * src_fps)) for k in range(n_out)]
        idx, frame, k = -1, None, 0
        f = _font(22)
        while k < n_out:
            while idx < want[k]:
                ok, frame_new = cap.read()
                if not ok:
                    k = n_out
                    break
                frame, idx = frame_new, idx + 1
            if k >= n_out or frame is None:
                break
            h, w = frame.shape[:2]
            s = min(W / w, (H - 50) / h)
            small = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            canvas = np.full((H, W, 3), BG[::-1], np.uint8)
            y0, x0 = (H - 50 - small.shape[0]) // 2, (W - small.shape[1]) // 2
            canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
            img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            d = ImageDraw.Draw(img)
            d.text(((W - d.textlength(caption, font=f)) / 2, H - 40), caption, font=f, fill=MUTED)
            bgr = cv2.cvtColor(np.asarray(self._stamp(img)), cv2.COLOR_RGB2BGR)
            self.proc.stdin.write(bgr.tobytes())
            self.frames += 1
            k += 1
        cap.release()


def _law_caption(t) -> str:
    bits = []
    for family, fits in t.fits:
        if fits:
            nxt = f" (next best ΔBIC {fits[1].delta_bic})" if len(fits) > 1 and fits[1].delta_bic else ""
            bits.append(f"{family}: {fits[0].name}{nxt}")
    return " · ".join(bits) or t.verdict


def build(session: Path, out: Path, clip_seconds: float, speed: float) -> tuple[Path, float]:
    nb = load(session)
    demo = Demo(nb, out)
    title = nb.title.replace("Kepler lab notebook: ", "").capitalize() or "Session"
    m = re.search(r"Scientist: ([\w.\-]+)", nb.setting)
    scientist = m[1].rstrip(".") if m else "the scientist model"
    tracking = ("an offline colour tracker stood in for SAM 3.1 (rehearsal)" if nb.mock_tracking
                else "SAM 3.1 tracked every frame")
    demo.card("Kepler · an autonomous desk lab", title,
              f"Session {session.name}. A camera is the instrument; {tracking}; {scientist} chose "
              "each experiment and analysed the data.", seconds=4.0)
    if nb.setting:
        demo.card(PROVENANCE[nb.provenance][0], "The setting", nb.setting, seconds=5.0)
    for t in nb.trials:
        demo.card(f"Trial {t.n}", t.heading, excerpt(t.why), seconds=6.0,
                  body_note="Why this trial, in the scientist's words:")
        if t.overlay and (session / t.overlay).exists():
            demo.clip(session / t.overlay, clip_seconds, speed,
                      f"Trial {t.n}: {t.heading}" + (f" · {speed:g}x" if speed != 1 else ""))
        if t.plot and (session / t.plot).exists():
            demo.still(session / t.plot, _law_caption(t), seconds=4.5)
    for f in nb.findings:
        demo.card(f"Finding · {f.confidence}", "", f.claim, seconds=5.0)
    demo.card("Where these numbers come from", f"sessions/{session.name}/NOTEBOOK.md",
              "Every decision, analysis and number in this video is in the session's lab notebook. "
              + PROVENANCE[nb.provenance][1], seconds=4.0)
    return out, demo.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session")
    p.add_argument("--out", help="default: <session>/demo.mp4")
    p.add_argument("--clip-seconds", type=float, default=6.0)
    p.add_argument("--speed", type=float, default=1.0, help="overlay playback speed")
    a = p.parse_args()
    session = Path(a.session)
    out, seconds = build(session, Path(a.out) if a.out else session / "demo.mp4", a.clip_seconds, a.speed)
    print(f"wrote {out} ({seconds:.0f} s)")


if __name__ == "__main__":
    main()
