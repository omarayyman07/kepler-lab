"""SAM 3.1 on Meta Model API, used as a measuring instrument.

One request per concept ("orange ball", "pendulum bob"). The response is
special-token text, one line per frame; Meta's `meta_sam_parser` turns it
into box and mask records. We reduce every mask to physical measurements:
an area-weighted sub-pixel centroid, the area, and the box, keyed by
SAM's stable object id.

Raw wire output is cached next to the clip, so re-analysis never pays twice.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from collections.abc import AsyncIterator, Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from meta_sam_parser import (
    decode_mask_to_raster,
    image_segmentation_format,
    parse_responses_stream,
    video_segmentation_format,
)

from . import config


@dataclass(frozen=True)
class Observation:
    frame: int
    object_id: str
    cx: float          # area-weighted centroid, source pixels
    cy: float
    area: float        # source pixels^2
    left: float
    top: float
    right: float
    bottom: float
    confidence: float | None


# --------------------------------------------------------------------------
# Parsing (shared by the live API, the cache, and the offline mock)
# --------------------------------------------------------------------------

def mask_to_observation(record) -> Observation:
    m = record.mask
    raster = np.frombuffer(decode_mask_to_raster(m), dtype=np.uint8).reshape(m.height, m.width)
    b = record.bounds
    sx = (b.right - b.left) / m.width
    sy = (b.bottom - b.top) / m.height
    rows, cols = np.nonzero(raster)
    if rows.size == 0:
        cx = (b.left + b.right) / 2
        cy = (b.top + b.bottom) / 2
        area = 0.0
    else:
        cx = b.left + (cols.mean() + 0.5) * sx
        cy = b.top + (rows.mean() + 0.5) * sy
        area = rows.size * sx * sy
    return Observation(
        frame=record.frame.frame_index if record.frame is not None else 0,
        object_id=str(record.object_id),
        cx=float(cx), cy=float(cy), area=float(area),
        left=float(b.left), top=float(b.top), right=float(b.right), bottom=float(b.bottom),
        confidence=record.confidence,
    )


async def _observations_from_events(events, *, video: bool) -> tuple[list[Observation], str]:
    fmt = video_segmentation_format() if video else image_segmentation_format()
    parsed = parse_responses_stream(events, fmt)
    result = await parsed.final_result()
    if getattr(result.outcome, "status", "completed") != "completed":
        print(f"[sam] warning: segmentation ended early: {result.outcome}")
    for d in result.diagnostics:
        print(f"[sam] diagnostic {d.severity} {d.code}: {d.message}")
    obs = [mask_to_observation(r) for r in result.records if r.kind == "mask"]
    obs.sort(key=lambda o: (o.frame, o.object_id))
    raw = getattr(result, "raw_output", "") or ""
    return obs, raw


async def events_from_text(text: str) -> AsyncIterator[dict]:
    """Replay cached or mock wire text as a well-formed Responses event stream."""
    lane = {"item_id": "msg_cached", "output_index": 0, "content_index": 0}
    for line in text.splitlines(keepends=True):
        if line.strip():
            yield {"type": "response.output_text.delta", "delta": line, **lane}
    yield {"type": "response.output_text.done", "text": text, **lane}
    yield {"type": "response.completed"}


def observations_from_text(text: str, *, video: bool = True) -> list[Observation]:
    obs, _ = asyncio.run(_observations_from_events(events_from_text(text), video=video))
    return obs


# --------------------------------------------------------------------------
# Live API
# --------------------------------------------------------------------------

def content_hash(path: Path) -> str:
    """Hash of the clip's bytes. lab.py rewrites the prepared clip on every run, so
    size+mtime would change each time and the paid SAM output would never be reused."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_path(clip: Path, concept: str) -> Path:
    h = hashlib.sha1(f"{content_hash(clip)}:{concept}".encode()).hexdigest()[:10]
    return clip.with_suffix(f".sam-{h}.txt")


def upload(path: str | Path) -> str:
    """Upload a clip once through the Files API; returns its file id."""
    with open(path, "rb") as f:
        return config.client().files.create(purpose="user_data", file=f).id


async def _segment(content: list[dict], *, video: bool,
                   metadata: dict | None = None) -> tuple[list[Observation], str]:
    client = config.async_client()
    events = await client.responses.create(
        model=config.SAM_MODEL,
        input=[{"type": "message", "role": "user", "content": content}],
        stream=True,
        metadata={"mask_encoding": "one_bit", "include_confidence": "true", **(metadata or {})},
    )
    return await _observations_from_events(events, video=video)


def _video_request(clip: Path, text: str, key: str, get_fid, use_cache: bool,
                   metadata: dict | None = None) -> list[Observation]:
    """One cached SAM video request. `text` is a noun phrase or object-prompt tokens."""
    cache = cache_path(clip, key)
    if use_cache and cache.exists():
        return observations_from_text(cache.read_text(), video=True)
    content = [
        {"type": "input_text", "text": text},
        {"type": "input_video", "file_id": get_fid()},
    ]
    obs, raw = asyncio.run(_segment(content, video=True, metadata=metadata))
    from .video import probe
    config.record_spend("sam", config.sam_cost(probe(clip).n_frames))
    # Cache even an empty answer: "SAM found nothing" was paid for too. A failed stream
    # raises before this point, so it is never cached.
    cache.write_text(raw or "")
    return obs


def box_prompt(o: Observation, frame: int, width: int, height: int, object_id: int = 0) -> str:
    """SAM object-prompt tokens: a box (media pixels, x2/y2 inclusive) on one frame.
    Same shape as an output line: frame marker, object id, then the box."""
    x1, y1 = max(0, int(np.floor(o.left))), max(0, int(np.floor(o.top)))
    x2, y2 = min(width - 1, int(np.ceil(o.right))), min(height - 1, int(np.ceil(o.bottom)))
    return f"<{frame}f>{object_id}<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={width};h={height}|>"


def backfill_start(clip: Path, obs: list[Observation], concept: str, get_fid,
                   use_cache: bool = True) -> list[Observation]:
    """The concept pass can start tracking late (on synthetic clips SAM 3.1 emitted
    nothing for frames 0-69). Re-prompt with the main object's box on its first
    frame and track backward to frame 0; merge by frame index under the same id."""
    if not obs:
        return obs
    counts: dict[str, int] = {}
    for o in obs:
        counts[o.object_id] = counts.get(o.object_id, 0) + 1
    target = max(counts, key=counts.get)
    first = min((o for o in obs if o.object_id == target), key=lambda o: o.frame)
    if first.frame == 0:
        return obs
    from .video import probe

    info = probe(clip)
    text = box_prompt(first, first.frame, info.width, info.height)
    back = _video_request(clip, text, f"{concept}|backward|{text}", get_fid, use_cache,
                          metadata={"propagation_direction": "backward"})
    # One box prompt should give one object; if SAM splits it, keep the one that
    # sits on the prompt box at the prompt frame.
    at_prompt = [o for o in back if o.frame == first.frame]
    if at_prompt:
        src = min(at_prompt, key=lambda o: np.hypot(o.cx - first.cx, o.cy - first.cy)).object_id
    elif back:
        c: dict[str, int] = {}
        for o in back:
            c[o.object_id] = c.get(o.object_id, 0) + 1
        src = max(c, key=c.get)
    else:
        print(f"[sam] backward pass from frame {first.frame} returned nothing")
        return obs
    early = [replace(o, object_id=target) for o in back if o.object_id == src and o.frame < first.frame]
    print(f"[sam] backward pass recovered {len(early)} of {first.frame} frames before frame {first.frame}")
    return sorted(early + obs, key=lambda o: (o.frame, o.object_id))


def segment_video(clip: str | Path, concept: str, *, file_id: str | None = None,
                  use_cache: bool = True, backfill: bool = True) -> list[Observation]:
    """Detect, segment and track every instance of `concept` in the clip.

    The clip is uploaded only if a request actually has to go out."""
    clip = Path(clip)
    fid = file_id

    def get_fid() -> str:
        nonlocal fid
        fid = fid or upload(clip)
        return fid

    obs = _video_request(clip, concept, concept, get_fid, use_cache)
    if backfill:
        obs = backfill_start(clip, obs, concept, get_fid, use_cache)
    return obs


def segment_image(image_bgr: np.ndarray, concept: str) -> list[Observation]:
    """Segment one still (used for calibration objects)."""
    import cv2

    ok, png = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    url = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()
    content = [
        {"type": "input_text", "text": concept},
        {"type": "input_image", "image_url": url},
    ]
    obs, _ = asyncio.run(_segment(content, video=False))
    return obs


def save_observations(obs: Iterable[Observation], path: str | Path) -> None:
    Path(path).write_text(json.dumps([asdict(o) for o in obs]))


def load_observations(path: str | Path) -> list[Observation]:
    return [Observation(**d) for d in json.loads(Path(path).read_text())]
