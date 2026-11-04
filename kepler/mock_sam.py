"""Offline stand-in for SAM 3.1, for tests and for working without credits.

It finds a strongly coloured object by HSV threshold and writes the result
in SAM 3.1's documented wire grammar, with masks encoded by Meta's own
`meta_sam_parser` codec. The real parsing path therefore runs unchanged:
the only thing mocked is the model.

It cannot do what SAM does (open-vocabulary concepts, clutter, occlusion,
identity across objects); it only exists so the pipeline is testable.
"""

from __future__ import annotations

import cv2
import numpy as np
from meta_sam_parser._mask_codec import _encode_raster  # Meta's encoder

# HSV ranges for the bob colours we use in synthetic and test clips.
COLOURS = {
    "orange": ((5, 120, 120), (25, 255, 255)),
    "red": ((0, 120, 100), (8, 255, 255)),
    "green": ((40, 80, 80), (85, 255, 255)),
    "blue": ((95, 120, 80), (130, 255, 255)),
}


def _record(object_id: int, mask: np.ndarray, w: int, h: int) -> str | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    crop = mask[y1 : y2 + 1, x1 : x2 + 1].astype(np.uint8)
    hh, ww = crop.shape
    payload = _encode_raster(crop.ravel().tolist(), ww, hh)  # starts with "!"
    box = f"<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={w};h={h};c=0.99|>"
    msk = f"<|mask;x=0;y=0;c=0.99;data={hh},{ww},{payload}|>"
    return f"{object_id}{box}{msk}"


def wire_text_for_video(path: str, colour: str = "orange", *, min_area: int = 30) -> str:
    lo, hi = COLOURS[colour]
    cap = cv2.VideoCapture(path)
    lines = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi)) > 0
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        if n > 1:
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            if stats[biggest, cv2.CC_STAT_AREA] >= min_area:
                rec = _record(0, labels == biggest, w, h)
                if rec:
                    lines.append(f"<{index}f>{rec}")
        index += 1
    cap.release()
    return "\n".join(lines) + ("\n" if lines else "")


def _observe(frame_bgr: np.ndarray, colour: str, index: int, min_area: int, soft: bool):
    """Largest blob of `colour` -> Observation.

    soft=True weights each pixel by its colour coverage (saturation above the local
    background) instead of a hard in/out threshold. Video is stored with 4:2:0 chroma,
    i.e. colour at half resolution, so hard colour-mask edges snap to a 2-pixel grid
    and the centroid picks up a position-dependent error: on synthetic drops it pulled
    the fitted drag exponent from 2 to 1.97 (2.5 sigma). A coverage-weighted centroid
    commutes with the codec's linear chroma resampling and is unbiased (n = 2.000).
    """
    from .sam import Observation

    lo, hi = COLOURS[colour]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lo), np.array(hi)) > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < min_area:
        return None
    blob = labels == k
    ys, xs = np.nonzero(blob)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    if not soft:
        return Observation(index, "0", float(xs.mean() + 0.5), float(ys.mean() + 0.5), float(xs.size),
                           float(x0), float(y0), float(x1), float(y1), 0.99)
    pad = 10
    sl = (slice(max(0, y0 - pad), y1 + pad), slice(max(0, x0 - pad), x1 + pad))
    b = blob[sl].astype(np.uint8)
    win = cv2.dilate(b, np.ones((7, 7), np.uint8)) > 0
    ring = (cv2.dilate(b, np.ones((15, 15), np.uint8)) > 0) & ~win
    s = hsv[..., 1][sl].astype(float)
    core = cv2.erode(b, np.ones((5, 5), np.uint8)) > 0
    bg = float(np.mean(s[ring])) if ring.any() else 0.0
    fg = float(np.mean(s[core])) if core.any() else float(np.mean(s[b > 0]))
    # Unclipped, so pixel noise averages out: interior weight 1 and background 0 on average.
    # Clipping at either end biases the area (the calibration scale) by up to ~1%.
    w = (s - bg) / max(fg - bg, 1.0) * win
    yy, xx = np.mgrid[sl[0].start:sl[0].start + w.shape[0], sl[1].start:sl[1].start + w.shape[1]]
    tot = float(w.sum())
    return Observation(index, "0", float((xx * w).sum() / tot + 0.5), float((yy * w).sum() / tot + 0.5),
                       tot, float(x0), float(y0), float(x1), float(y1), 0.99)


def observations_for_image(image_bgr: np.ndarray, colour: str = "green", *, min_area: int = 30,
                           soft: bool = True):
    """Still-image stand-in for `sam.segment_image` (calibration objects)."""
    o = _observe(image_bgr, colour, 0, min_area, soft)
    return [o] if o else []


def observations_for_video(path: str, colour: str = "orange", *, min_area: int = 30, soft: bool = True):
    """Fast path: same detection without the wire encoding round trip, and with
    coverage-weighted centroids (see `_observe`). `soft=False` reproduces exactly what
    `wire_text_for_video` encodes (binary masks, like SAM's one_bit output)."""
    cap = cv2.VideoCapture(path)
    out = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        o = _observe(frame, colour, index, min_area, soft)
        if o:
            out.append(o)
        index += 1
    cap.release()
    return out
