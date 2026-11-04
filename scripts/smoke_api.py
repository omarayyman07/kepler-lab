"""Live smoke test of every Meta Model API call Kepler makes.

    python scripts/smoke_api.py [--seconds 5]

1. list models (are sam-3.1 and muse-spark-1.3 visible to this key?)
2. render a short synthetic clip with known physics and upload it once
3. SAM 3.1 tracks "orange ball"; centroids are checked against the offline
   colour tracker on the same frames
4. Muse Spark QC watches the clip and returns its verdict

Cost: one SAM request over seconds*60 frames, one Muse Spark video request.
Both are cached beside the clip, so a re-run is free.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kepler import config, mock_sam, qc, sam, synthetic, video  # noqa: E402


def step(name: str):
    print(f"\n--- {name}")
    return time.perf_counter()


def scientist_probe() -> bool:
    """One real planning step of the agent loop: does the model call run_trial with valid args?"""
    from kepler import agent
    from kepler.experiments import PendulumExperiment

    exp = PendulumExperiment(length_m=0.8)
    msgs = [{"role": "system", "content": agent.system_prompt(exp, budget=4, releaser="human assistant")},
            {"role": "user", "content": "The lab is ready. Plan your first trial."}]
    try:
        resp = config.client().chat.completions.create(
            model=config.SCIENTIST_MODEL, messages=msgs, tools=agent.tools(exp),
            max_tokens=16000, reasoning_effort="high",
        )
    except Exception as e:  # noqa: BLE001
        print(f"FAILED: {type(e).__name__}: {e}")
        return False
    msg = resp.choices[0].message
    usage = getattr(resp, "usage", None)
    if usage is not None:
        print(f"tokens: prompt {usage.prompt_tokens}, completion {usage.completion_tokens}")
    if not msg.tool_calls:
        print(f"no tool call; content: {(msg.content or '')[:300]!r}")
        return False
    for call in msg.tool_calls:
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError as e:
            print(f"{call.function.name}: bad JSON arguments: {e}")
            return False
        print(f"{call.function.name}({json.dumps(args)[:300]})")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--out", default="sessions/smoke")
    a = p.parse_args()
    out = Path(a.out)

    t0 = step("models")
    ids = sorted(m.id for m in config.client().models.list())
    print(", ".join(ids))
    for need in (config.SAM_MODEL, config.QC_MODEL):
        print(f"{need}: {'listed' if need in ids else 'NOT LISTED'}")
    print(f"({time.perf_counter() - t0:.1f}s)")

    t0 = step("synthetic clip")
    truth = synthetic.PendulumTruth(seconds=a.seconds, fps=60)
    synthetic.render(out / "raw.mp4", truth)
    clip = video.prepare_clip(out / "raw.mp4", out / "clip.mp4")
    info = video.probe(clip)
    print(f"{clip}: {info.n_frames} frames, {info.width}x{info.height}, {info.fps:.2f} fps")

    phrase = "orange ball"
    experiment = "a simple pendulum swinging freely"
    sam_cached = sam.cache_path(clip, phrase).exists()
    qc_cache = qc.cache_path(clip, experiment)
    file_id = None
    if not (sam_cached and qc_cache.exists()):
        t0 = step("upload")
        file_id = sam.upload(clip)
        print(f"file id {file_id} ({time.perf_counter() - t0:.1f}s)")

    t0 = step(f"SAM 3.1 '{phrase}'" + (" (cached)" if sam_cached else ""))
    obs = sam.segment_video(clip, phrase, file_id=file_id)
    print(f"{len(obs)} masks, objects {sorted({o.object_id for o in obs})}, "
          f"frames {min(o.frame for o in obs) if obs else '-'}..{max(o.frame for o in obs) if obs else '-'} "
          f"({time.perf_counter() - t0:.1f}s)")
    ref = {o.frame: o for o in mock_sam.observations_for_video(str(clip))}
    d = np.array([np.hypot(o.cx - ref[o.frame].cx, o.cy - ref[o.frame].cy) for o in obs if o.frame in ref])
    if d.size:
        print(f"centroid vs colour tracker: median {np.median(d):.2f}px, max {d.max():.2f}px "
              f"over {d.size}/{len(ref)} frames")
    conf = [o.confidence for o in obs if o.confidence is not None]
    if conf:
        print(f"confidence {min(conf):.2f}..{max(conf):.2f}")

    t0 = step("Muse Spark QC" + (" (cached)" if qc_cache.exists() else ""))
    rep = qc.check_clip(file_id, experiment, cache=qc_cache)
    print(f"usable={rep.usable} track='{rep.track_phrase}' issues={rep.issues}")
    print(f"notes: {rep.notes}")
    if not rep.track_phrase:
        print(f"raw reply: {rep.raw[:500]!r}")
    print(f"({time.perf_counter() - t0:.1f}s)")

    t0 = step(f"scientist tool call ({config.SCIENTIST_MODEL})")
    sci_ok = scientist_probe()
    print(f"({time.perf_counter() - t0:.1f}s for one planning step)")

    ok = bool(obs) and d.size > 0.9 * len(ref) and rep.track_phrase is not None and sci_ok
    print("\nSMOKE " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
