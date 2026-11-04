"""Remove camera motion by tracking a wall-fixed feature at the pivot.

A handheld phone drifts by tens to hundreds of pixels over a clip. The
pendulum analysis assumes a fixed pivot, so that drift goes straight into the
angles (and into a ruined circle fit). Angles are invariant to scale, so
removing translation is enough to first order.

The pivot hardware (nail, knot, clamp) is the one textured object guaranteed
to sit still on the wall next to the pendulum, at the same depth. We:

  1. find a seed: the pivot from a circle fit over the short window of the raw
     track that is most circular (camera motion is small over a few seconds);
  2. cut a template from the region at and above the seed (the cord hangs
     below it, so it mostly stays out of the template);
  3. match it in every frame (normalised cross-correlation, sub-pixel peak),
     gate low scores and jumps, interpolate short gaps.

Frames where the feature cannot be found are reported, and the bob
observations in them are dropped rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .sam import Observation


@dataclass
class CameraMotion:
    offsets: np.ndarray      # (n_frames, 2) feature position minus its reference position, px
    valid: np.ndarray        # (n_frames,) bool
    ref_frame: int
    seed: tuple[float, float]
    scores: np.ndarray

    @property
    def range_px(self) -> tuple[float, float]:
        o = self.offsets[self.valid]
        return (float(np.ptp(o[:, 0])), float(np.ptp(o[:, 1]))) if o.size else (np.nan, np.nan)

    def summary(self) -> str:
        rx, ry = self.range_px
        return (f"camera moved {rx:.0f}px horizontally and {ry:.0f}px vertically; "
                f"pivot feature found in {self.valid.mean():.0%} of frames")


def _seed(x, y, t, window_s: float = 3.0, min_span_deg: float = 15.0):
    """Pivot and reference time from the most circular short window of the raw track."""
    from .pendulum import fit_circle

    best = None
    for t0 in np.arange(t[0], t[-1] - window_s, window_s / 2):
        m = (t >= t0) & (t < t0 + window_s)
        if m.sum() < 20:
            continue
        xc, yc, r, rms = fit_circle(x[m], y[m])
        span = np.degrees(np.ptp(np.arctan2(x[m] - xc, y[m] - yc)))
        if span < min_span_deg or not np.isfinite(rms):
            continue
        if best is None or rms < best[0]:
            best = (rms, xc, yc, t0 + window_s / 2)
    return best


def _pivot_feature(gray: np.ndarray, bob: tuple[float, float], pivot: tuple[float, float],
                   *, contrast: float = 35.0, band_px: float = 15.0,
                   dark: bool = False) -> tuple[float, float] | None:
    """Top end of the string: follow the high-contrast structure on the line bob -> pivot upward.

    A circle fit over a short arc fixes the pivot well sideways but poorly along the
    string, so we look for the hardware (nail, knot, clamp) where the string ends.
    `dark=True` follows only structure darker than the wall (a dark string), so a
    light object touching the knot (a tape measure) cannot hijack it."""
    H, W = gray.shape
    bx, by = bob
    px, py = pivot
    x0 = int(max(min(bx, px) - 120, 0)); x1 = int(min(max(bx, px) + 120, W))
    y0 = int(max(py - 0.5 * (by - py), 0)); y1 = int(min(by - 0.1 * (by - py), H))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    box = gray[y0:y1, x0:x1].astype(np.float32)
    diff = np.median(box) - box if dark else np.abs(box - np.median(box))
    mask = (diff > contrast).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels = cv2.connectedComponents(mask)
    if n < 2:
        return None
    ys, xs = np.mgrid[y0:y1, x0:x1]
    # distance of each pixel from the (extended) line through bob and pivot
    dx, dy = px - bx, py - by
    dist = np.abs(dy * (xs - bx) - dx * (ys - by)) / np.hypot(dx, dy)
    near = dist < band_px
    counts = np.bincount(labels[near], minlength=n)
    counts[0] = 0
    k = int(np.argmax(counts))
    if counts[k] < 20:
        return None
    rows, cols = np.nonzero(labels == k)
    top = rows.min()
    sel = rows <= top + 4
    return float(x0 + np.median(cols[sel])), float(y0 + top)


def trace_string_top(gray: np.ndarray, bob: tuple[float, float], length_px: float, *,
                     contrast: float = 35.0, start_frac: float = 0.25, window: int = 6,
                     max_miss: int = 4) -> tuple[float, float] | None:
    """Follow a dark string upward from just above the bob, row by row, using its own slope;
    return where it ends (the top of the knot or nail). `length_px` is a rough string length."""
    H, W = gray.shape
    g = gray.astype(np.float32)
    bx, by = bob
    y = int(by - start_frac * length_px)
    if not 0 < y < H:
        return None
    row = g[y, max(int(bx) - 60, 0):min(int(bx) + 60, W)]
    bg = np.median(row)
    dark = np.nonzero(bg - row > contrast)[0]
    if dark.size == 0:
        return None
    x = max(int(bx) - 60, 0) + float(np.median(dark))
    xs, ys = [x], [y]
    miss = 0
    while y > 1:
        y -= 1
        slope = np.polyfit(ys[-40:], xs[-40:], 1)[0] if len(ys) > 10 else 0.0
        xp = xs[-1] + slope * (y - ys[-1])
        lo, hi = int(xp) - window, int(xp) + window + 1
        if lo < 0 or hi > W:
            break
        seg = g[y, lo:hi]
        bgl = np.median(g[y, max(lo - 30, 0):min(hi + 30, W)])
        d = np.nonzero(bgl - seg > contrast)[0]
        if d.size == 0:
            miss += 1
            if miss > max_miss:
                break
            continue
        miss = 0
        xs.append(lo + float(np.mean(d)))
        ys.append(y)
    if len(ys) < 0.3 * length_px:          # lost the string well before reaching the top
        return None
    return float(np.median(xs[-3:])), float(ys[-1])


def string_top(clip, track_x, track_y, track_frames, theta, near: tuple[float, float], *,
               offsets: np.ndarray | None = None, max_angle_deg: float = 3.0,
               n_samples: int = 40) -> tuple[float, float, float] | None:
    """Where a dark string ends at the top, measured offline in frames where the bob hangs
    near the bottom (string close to vertical). Returns the median (x, y) and the rms scatter
    (px) over the sampled frames: an independent measurement of the pivot, no circle fit."""
    idx = np.nonzero(np.abs(theta) < np.radians(max_angle_deg))[0]
    if idx.size == 0:
        return None
    idx = idx[np.linspace(0, idx.size - 1, min(n_samples, idx.size)).astype(int)]
    cap = cv2.VideoCapture(str(clip))
    pts = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(track_frames[i]))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # track coordinates may be camera-stabilised; the picture is not
        ox, oy = offsets[int(track_frames[i])] if offsets is not None else (0.0, 0.0)
        bob = (track_x[i] + ox, track_y[i] + oy)
        p = trace_string_top(gray, bob, float(np.hypot(bob[0] - near[0] - ox, bob[1] - near[1] - oy)))
        if p is not None:
            pts.append((p[0] - ox, p[1] - oy))
    cap.release()
    if len(pts) < 5:
        return None
    pts = np.array(pts)
    med = np.median(pts, axis=0)
    rms = float(np.sqrt(np.mean(np.sum((pts - med) ** 2, axis=1))))
    return float(med[0]), float(med[1]), rms


def _subpixel(r: np.ndarray, i: int, j: int) -> tuple[float, float]:
    di = dj = 0.0
    if 0 < i < r.shape[1] - 1:
        a, b, c = r[j, i - 1], r[j, i], r[j, i + 1]
        den = a - 2 * b + c
        di = 0.5 * (a - c) / den if den else 0.0
    if 0 < j < r.shape[0] - 1:
        a, b, c = r[j - 1, i], r[j, i], r[j + 1, i]
        den = a - 2 * b + c
        dj = 0.5 * (a - c) / den if den else 0.0
    return di, dj


def track_pivot(clip, track_x, track_y, track_t, timestamps, *, half_w: int = 60, above: int = 40,
                below: int = 40, search: int = 220, min_score: float = 0.6, max_jump_px: float = 5.0,
                max_gap: int = 6, min_texture: float = 8.0, window_s: float = 3.0) -> CameraMotion | None:
    """Track the wall-fixed feature at the pivot through the clip. None if there is no usable feature."""
    seed = _seed(track_x, track_y, track_t, window_s)
    if seed is None:
        return None
    _, cx, cy, t_mid = seed
    # reference frame: inside the seed window, the bob nearest the bottom (string near vertical)
    win = np.nonzero(np.abs(track_t - t_mid) <= window_s / 2)[0]
    i_ref = win[np.argmin(np.abs(track_x[win] - cx))]
    ref = int(np.argmin(np.abs(timestamps - track_t[i_ref])))
    cap = cv2.VideoCapture(str(clip))
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    feat = _pivot_feature(gray, (track_x[i_ref], track_y[i_ref]), (cx, cy))
    if feat is None:
        cap.release()
        return None
    sx, sy = feat
    x0, x1 = max(int(round(sx)) - half_w, 0), min(int(round(sx)) + half_w, W)
    y0, y1 = max(int(round(sy)) - above, 0), min(int(round(sy)) + below, H)
    if x1 - x0 < 24 or y1 - y0 < 24:
        cap.release()
        return None
    tpl = gray[y0:y1, x0:x1].astype(np.float32)
    if tpl.std() < min_texture:   # a blank wall: nothing to lock on to
        cap.release()
        return None

    # Search a fixed wide window around the reference position in every frame: no
    # frame-to-frame following, so one bad match cannot drag the tracker away.
    wx0, wy0 = max(x0 - search, 0), max(y0 - search, 0)
    wx1, wy1 = min(x1 + search, W), min(y1 + search, H)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    pos, scores = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        win = cv2.cvtColor(frame[wy0:wy1, wx0:wx1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        r = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, (i, j) = cv2.minMaxLoc(r)
        di, dj = _subpixel(r, i, j)
        # A peak on the search window's edge may be cut off, unless that edge is the image edge.
        on_edge = ((i == 0 and wx0 > 0) or (i == r.shape[1] - 1 and wx1 < W)
                   or (j == 0 and wy0 > 0) or (j == r.shape[0] - 1 and wy1 < H))
        pos.append((wx0 + i + di - x0, wy0 + j + dj - y0))
        scores.append(mx if not on_edge else 0.0)
    cap.release()
    off = np.array(pos, float)
    scores = np.array(scores)
    n = len(off)
    valid = scores >= min_score
    # jumps: compare with a running median of the valid track
    if valid.sum() > 10:
        idx = np.nonzero(valid)[0]
        for k in range(2):
            med = np.array([np.median(off[idx[max(0, q - 7):q + 8], k]) for q in range(idx.size)])
            bad = np.abs(off[idx, k] - med) > max_jump_px
            valid[idx[bad]] = False
    # interpolate short gaps; long gaps stay invalid
    if valid.sum() >= 2:
        good = np.nonzero(valid)[0]
        filled = valid.copy()
        for a, b in zip(good[:-1], good[1:]):
            if 1 < b - a <= max_gap + 1:
                for k in range(2):
                    off[a + 1:b, k] = np.interp(np.arange(a + 1, b), [a, b], off[[a, b], k])
                filled[a + 1:b] = True
        valid = filled
    return CameraMotion(off[:n], valid[:n], ref, (float(sx), float(sy)), scores)


def apply(obs: list[Observation], motion: CameraMotion) -> list[Observation]:
    """Bob positions in the frame of the pivot feature; observations without a valid offset are dropped."""
    out = []
    for o in obs:
        if o.frame >= len(motion.valid) or not motion.valid[o.frame]:
            continue
        dx, dy = motion.offsets[o.frame]
        out.append(Observation(o.frame, o.object_id, o.cx - dx, o.cy - dy, o.area,
                               o.left - dx, o.top - dy, o.right - dx, o.bottom - dy, o.confidence))
    return out
