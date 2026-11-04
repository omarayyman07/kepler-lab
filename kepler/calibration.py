"""Metres per pixel from a SAM segment of a reference object of known size.

The reference must sit in the plane of the motion (same distance from the
camera), or the scale is wrong by the ratio of the distances.

Sizes come from the mask, not the box, where possible: for a round object the
area-equivalent diameter 2*sqrt(area/pi) averages over the whole outline and
is sub-pixel; box edges are only good to about a pixel each.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .sam import Observation, content_hash


@dataclass(frozen=True)
class Reference:
    phrase: str              # what SAM segments, e.g. "green disc", "ruler"
    size_m: float            # the known size
    kind: str = "diameter"   # diameter (round, from mask area) | height | width (from box)


@dataclass
class Calibration:
    m_per_px: float
    size_px: float
    reference: Reference
    frame: int
    observation: Observation


def size_px(o: Observation, kind: str) -> float:
    if kind == "diameter":
        return 2.0 * float(np.sqrt(o.area / np.pi))
    if kind == "height":
        return o.bottom - o.top
    if kind == "width":
        return o.right - o.left
    raise ValueError(f"unknown reference kind {kind!r}")


def calibrate(clip: Path, ref: Reference, segment_image, *, frame: int = 0,
              use_cache: bool = True) -> Calibration:
    """Segment `ref.phrase` on one still and convert its known size to a scale.

    Live SAM output is cached beside the clip (`*.cal-*.json`): an image request costs money."""
    from .video import read_frame

    clip = Path(clip)
    key = hashlib.sha1(f"{content_hash(clip)}:{ref.phrase}:{frame}".encode()).hexdigest()[:10]
    cache = clip.with_suffix(f".cal-{key}.json")
    if use_cache and cache.exists():
        obs = [Observation(**d) for d in json.loads(cache.read_text())]
    else:
        obs = segment_image(read_frame(clip, frame), ref.phrase)
        if use_cache:
            cache.write_text(json.dumps([asdict(o) for o in obs]))
    if not obs:
        raise ValueError(f"Calibration: '{ref.phrase}' not found on frame {frame}.")
    # Several matches: trust the most confident, then the largest.
    best = max(obs, key=lambda o: (o.confidence or 0.0, o.area))
    px = size_px(best, ref.kind)
    if px < 20:
        raise ValueError(f"Calibration: '{ref.phrase}' is only {px:.1f}px across; "
                         "use a bigger reference or move the camera closer.")
    return Calibration(ref.size_m / px, px, ref, frame, best)
