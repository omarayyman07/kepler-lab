"""Reproduce a published measurement from its cached SAM output: no video, no API key.

    python -m kepler replay results/<name>

A results folder holds `replay.json` (what the live run measured besides SAM: frame
times, frame size, the camera-drift track if one was computed, the pivot measured on
the picture, the plugin's parameters, and the published numbers) and `sam/`, the raw
SAM 3.1 wire output exactly as the API streamed it. Replay parses that output with
Meta's `meta_sam_parser`, merges the backfill pass the same way the live run does, and
runs the same physics. It then compares every published number with the new one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import lab, sam
from .experiments import EXPERIMENTS, TrialContext


def load_observations(folder: Path, data: dict) -> list[sam.Observation]:
    caches = data["sam_caches"]
    fwd = next(c for c in caches if "|backward|" not in c["key"])
    obs = sam.observations_from_text((folder / "sam" / fwd["file"]).read_text(), video=True)
    back = next((c for c in caches if "|backward|" in c["key"]), None)
    if back is not None:
        obs = sam.merge_backfill(obs, sam.observations_from_text((folder / "sam" / back["file"]).read_text(),
                                                                 video=True))
    return obs


def replay(folder: str | Path):
    folder = Path(folder)
    data = json.loads((folder / "replay.json").read_text(encoding="utf-8"))
    exp = EXPERIMENTS[data["experiment"]](**data["params"])
    obs = load_observations(folder, data)
    ts = np.array(data["timestamps"], float)
    ctx = TrialContext(clip=folder / "<no video>", n_frames=int(data["n_frames"]),
                       control_value=data.get("control_value"), segment_image=None, use_cache=False,
                       frame_size=tuple(data["frame_size"]), replay=data)
    m = exp.measure(obs, ts, ctx)
    failures = lab.tracking_failures(obs, int(data["n_frames"])) + m.evidence_failures
    return data, m, failures


def compare(published: dict, new: dict, rel: float = 1e-9) -> list[str]:
    """Names of published headline numbers that did not come out the same."""
    bad = []
    for k, v in published.items():
        w = new.get(k)
        if isinstance(v, (int, float)) and v == v:   # skip NaN
            if w is None or abs(w - v) > rel * max(abs(v), 1e-12):
                bad.append(f"{k}: published {v!r}, replay {w!r}")
    return bad
