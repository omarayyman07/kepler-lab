"""Visual outputs: a measurement overlay video (for the demo) and plots."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from .pendulum import PendulumRun

ACCENT = (0, 200, 255)     # BGR amber
TRAIL = (255, 180, 60)
TEXT = (255, 255, 255)
MEASURED = (80, 220, 80)   # BGR green: the pivot measured on the picture


def _text_panel(img: np.ndarray, lines: list[str], scale: float) -> None:
    """Text on a translucent dark panel, drawn once. (An outline drawn with a thicker
    stroke drifts sideways from the text, because OpenCV advances wider glyphs further.)"""
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale
    th = max(1, int(round(2 * scale)))
    step = int(30 * scale)
    pad = int(10 * scale)
    width = max(cv2.getTextSize(s, font, fs, th)[0][0] for s in lines)
    x0, y0 = int(20 * scale), int(36 * scale)
    top, bottom = y0 - step + pad // 2, y0 + step * (len(lines) - 1) + pad
    roi = img[max(top - pad, 0):bottom, max(x0 - pad, 0):x0 + width + pad]
    roi[:] = (roi * 0.45).astype(img.dtype)
    for k, s in enumerate(lines):
        cv2.putText(img, s, (x0, y0 + k * step), font, fs, ACCENT if k == 0 else TEXT, th, cv2.LINE_AA)


def overlay_video(clip: str | Path, run: PendulumRun, out: str | Path, *, title: str = "KEPLER",
                  g_text: str | None = None, trail: int = 25) -> Path:
    clip, out = Path(clip), Path(out)
    tmp = out.with_suffix(".tmp.mp4")
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    by_frame = {int(f): i for i, f in enumerate(run.track.frames)}
    # Positions are in the camera-stabilised frame; draw them back on the moving picture.
    offsets = run.camera.offsets if run.camera is not None else None
    cyc = run.cycles
    idx = 0
    pts: list[tuple[float, float]] = []   # stabilised (wall) coordinates
    scale = max(0.6, w / 1280)
    while True:
        ok, img = cap.read()
        if not ok:
            break
        ox, oy = offsets[idx] if offsets is not None and idx < len(offsets) else (0.0, 0.0)
        px, py = int(round(run.pivot[0] + ox)), int(round(run.pivot[1] + oy))
        on_screen = [(int(round(a + ox)), int(round(b + oy))) for a, b in pts]
        i = by_frame.get(idx)
        if i is not None:
            pts.append((run.track.x[i], run.track.y[i]))
            pts = pts[-trail:]
            on_screen = [(int(round(a + ox)), int(round(b + oy))) for a, b in pts]
            t = run.track.t[i]
            cv2.line(img, (px, py), on_screen[-1], ACCENT, 2, cv2.LINE_AA)
            for a, b in zip(on_screen[:-1], on_screen[1:]):
                cv2.line(img, a, b, TRAIL, 2, cv2.LINE_AA)
            cv2.circle(img, on_screen[-1], int(6 * scale), ACCENT, 2, cv2.LINE_AA)
            # equilibrium line
            ex = px + int(run.radius_px * np.sin(run.theta_eq))
            ey = py + int(run.radius_px * np.cos(run.theta_eq))
            cv2.line(img, (px, py), (ex, ey), (180, 180, 180), 1, cv2.LINE_AA)
            done = cyc.period[cyc.t_mid + cyc.period / 2 <= t] if cyc.period.size else []
            lines = [title, f"t = {t:6.2f} s", f"theta = {np.degrees(run.theta[i]):+6.1f} deg"]
            if len(done):
                lines.append(f"last period = {done[-1]*1000:7.1f} ms   (cycle {len(done)})")
            if g_text:
                lines.append(g_text)
            _text_panel(img, lines, scale)
        if run.fit_pivot is not None:   # circle-fit centre, for comparison with the measured pivot
            fx, fy = int(round(run.fit_pivot[0] + ox)), int(round(run.fit_pivot[1] + oy))
            r = int(9 * scale)
            cv2.line(img, (fx - r, fy - r), (fx + r, fy + r), (70, 70, 70), 2, cv2.LINE_AA)
            cv2.line(img, (fx - r, fy + r), (fx + r, fy - r), (70, 70, 70), 2, cv2.LINE_AA)
        cv2.circle(img, (px, py), int(5 * scale), ACCENT, -1, cv2.LINE_AA)
        if run.measured_pivot is not None:   # the pivot as measured on the picture (ring)
            mx, my = int(round(run.measured_pivot[0] + ox)), int(round(run.measured_pivot[1] + oy))
            cv2.circle(img, (mx, my), int(9 * scale), MEASURED, 2, cv2.LINE_AA)
        vw.write(img)
        idx += 1
    cap.release()
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(out)], check=True)
    tmp.unlink(missing_ok=True)
    return out


def plot_run(run: PendulumRun, g_fits, decay_fits, L_eff: float, out_png: str | Path,
             tracker: str = "SAM 3.1") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .pendulum import decay_models, period_models

    c = run.cycles
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].plot(run.track.t, np.degrees(run.theta), lw=0.8)
    ax[0].set(xlabel="time (s)", ylabel="angle (deg)", title=f"Measured angle ({tracker} track)")

    A = np.degrees(c.amplitude)
    ax[1].plot(A, c.period * 1000, "o", ms=4, label="measured cycles")
    if c.amplitude.size:
        aa = np.linspace(0, max(c.amplitude.max() * 1.1, 0.2), 200)
        models = {m.name: m for m in period_models(L_eff)}
        for f in g_fits:
            if f.ok:
                ax[1].plot(np.degrees(aa), models[f.name].func(aa, *f.params.values()) * 1000,
                           label=f"{f.name} (ΔBIC {f.delta_bic:.0f})")
    ax[1].set(xlabel="amplitude (deg)", ylabel="period (ms)", title="Period vs amplitude")
    ax[1].legend(fontsize=7)

    tt = c.t_mid - c.t_mid[0] if c.t_mid.size else c.t_mid
    ax[2].plot(tt, A, "o", ms=4, label="measured")
    if tt.size:
        ts = np.linspace(0, tt.max(), 200)
        models = {m.name: m for m in decay_models()}
        for f in decay_fits:
            if f.ok:
                ax[2].plot(ts, np.degrees(models[f.name].func(ts, *f.params.values())),
                           label=f"{f.name} (ΔBIC {f.delta_bic:.0f})")
    ax[2].set(xlabel="time (s)", ylabel="amplitude (deg)", title="Amplitude decay")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return Path(out_png)
