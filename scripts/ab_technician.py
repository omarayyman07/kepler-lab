"""A/B test the technician model on clips whose correct verdict is known.

    python scripts/ab_technician.py [--runs 5] [--models muse-spark-1.3 muse-wonder-1.0]

clean: one pendulum, nothing else             -> correct verdict: usable
bad:   a second orange ball rolls through the
       bottom of the swing mid-clip            -> correct verdict: reject

Every call goes to the API (no cache): we are measuring the model's variance.
An "unchecked" reply (empty/unparsable after a retry) counts as wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kepler import config, qc, sam, synthetic, video  # noqa: E402

EXPERIMENT = "a simple pendulum swinging freely"
MAX_USD = 3.0


def make_clips(out: Path) -> dict[str, Path]:
    clips = {}
    for name, intruder in (("clean", None), ("bad", (0.35, 0.75))):
        raw = out / f"{name}.raw.mp4"
        synthetic.render(raw, synthetic.PendulumTruth(seconds=5, fps=60, intruder=intruder))
        clips[name] = video.prepare_clip(raw, out / f"{name}.mp4")
    return clips


def cost(model: str, usage: dict) -> float:
    p_in, _, p_out = config.PRICES[model]
    return (usage.get("input_tokens", 0) * p_in + usage.get("output_tokens", 0) * p_out) / 1e6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--models", nargs="+", default=["muse-spark-1.3", "muse-wonder-1.0"])
    p.add_argument("--out", default="sessions/ab_technician")
    a = p.parse_args()
    out = Path(a.out)
    clips = make_clips(out)
    fids = {name: sam.upload(c) for name, c in clips.items()}
    print(f"uploaded {fids}")

    rows, spent = [], 0.0
    for model in a.models:
        for name, fid in fids.items():
            for i in range(a.runs):
                if spent > MAX_USD:
                    sys.exit(f"stopping: spent ${spent:.2f} > ${MAX_USD}")
                t0 = time.perf_counter()
                rep = qc.check_clip(fid, EXPERIMENT, model=model)
                dt = time.perf_counter() - t0
                c = cost(model, rep.usage)
                spent += c
                correct = rep.checked and rep.usable == (name == "clean")
                rows.append({"model": model, "clip": name, "run": i + 1, "checked": rep.checked,
                             "usable": rep.usable, "correct": correct, "seconds": round(dt, 1),
                             "usd": round(c, 5), "usage": rep.usage, "issues": rep.issues,
                             "track_phrase": rep.track_phrase})
                print(f"{model:16s} {name:5s} #{i + 1}  {'OK ' if correct else 'BAD'} "
                      f"usable={rep.usable} checked={rep.checked} {dt:5.1f}s ${c:.4f}  {rep.issues}",
                      flush=True)
    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\nmodel             clean ok   bad ok   unchecked   mean s   $/call")
    for model in a.models:
        r = [x for x in rows if x["model"] == model]
        ok = lambda clip: sum(x["correct"] for x in r if x["clip"] == clip)
        n = lambda clip: sum(1 for x in r if x["clip"] == clip)
        print(f"{model:16s}  {ok('clean')}/{n('clean')}        {ok('bad')}/{n('bad')}      "
              f"{sum(not x['checked'] for x in r)}           "
              f"{sum(x['seconds'] for x in r) / len(r):5.1f}   {sum(x['usd'] for x in r) / len(r):.4f}")
    print(f"total ${spent:.3f}; rows in {out / 'results.json'}")


if __name__ == "__main__":
    main()
