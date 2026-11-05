"""Kepler Lab Replay: a static web page that replays real trials with the analysis drawn live.

    python -m kepler showcase [--out docs/showcase] [--feed]

For each published result (results/real1, results/tomato1) this writes a compressed
H.264 copy of the clip and one data JSON holding:

  * the overlay track per frame: bob centroid and mask outline (decoded from the raw
    SAM 3.1 output in results/<name>/sam), the circle-fit centre and the measured pivot,
    θ, all in the clip's pixel coordinates (camera motion added back where the analysis
    removed it)
  * causal snapshots, one per completed cycle: every period and damping law re-fitted
    on the cycles completed so far, ΔBIC, the leading law, g ± σ, and the diagnosis
    evidence (implied length, free coefficient, pivot offset, mask jumps) as known then
  * the live feed: commentary written once by Muse Wonder from those snapshots, kept
    only if every number in it appears in the data it was given

Nothing is smoothed: the last snapshot is the published result (checked here).
The page itself (index.html, app.js, style.css, vendored KaTeX) is static.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import config, pendulum, sam
from .replay import compare, replay

ROOT = Path(__file__).resolve().parents[1]
PAGE_SRC = Path(__file__).with_name("showcase_page")
G_LOCAL = 9.79          # Cairo, for the implied-length diagnosis only
MIN_EVIDENCE_CYCLES = 5  # evidence statements and diagnosis cards need at least this many cycles
LAW_IDS = {"isochronous (small-angle)": "H1", "quadratic correction": "H2",
           "exact (elliptic integral)": "H3", "free quadratic": "H4"}
DAMP_IDS = {"linear (viscous) damping": "D1", "quadratic (air drag) damping": "D2",
            "dry (Coulomb) friction": "D3", "viscous + air drag": "D4"}

TRIALS = [
    {"id": "keys", "result": "results/real1", "clip": "sessions/real1/trial01.mp4",
     "title": "Keys on a ring, thick doubled cord", "short": "Keys",
     "setup": "bunch of keys on a doubled cord, handheld phone", "colour": "#5aa9ff"},
    {"id": "shoelace", "result": "results/tomato1", "clip": "sessions/tomato1/trial01.mp4",
     "title": "Tomato on a heavy shoelace", "short": "Shoelace",
     "setup": "tomato in a plastic bag on a shoelace, phone about 14° off square", "colour": "#f0a050"},
]


# --------------------------------------------------------------------------
# Mask outlines from the raw SAM output
# --------------------------------------------------------------------------

def _records(text: str):
    from meta_sam_parser import parse_responses_stream, video_segmentation_format

    async def go():
        res = parse_responses_stream(sam.events_from_text(text), video_segmentation_format())
        return (await res.final_result()).records
    return [r for r in asyncio.run(go()) if r.kind == "mask"]


def _outline(record) -> tuple[list[float], tuple[float, float]] | None:
    from meta_sam_parser import decode_mask_to_raster

    m, b = record.mask, record.bounds
    raster = np.frombuffer(decode_mask_to_raster(m), dtype=np.uint8).reshape(m.height, m.width)
    cs, _ = cv2.findContours((raster > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 0.8, True)[:, 0, :].astype(float)
    sx, sy = (b.right - b.left) / m.width, (b.bottom - b.top) / m.height
    xs, ys = b.left + (c[:, 0] + 0.5) * sx, b.top + (c[:, 1] + 0.5) * sy
    pts = [round(float(v), 1) for xy in zip(xs, ys) for v in xy]
    return pts, (float(np.mean(xs)), float(np.mean(ys)))


def outlines(folder: Path, data: dict, obs_by_frame: dict[int, sam.Observation]) -> dict[int, list[float]]:
    """Frame -> mask outline of the tracked bob (forward pass first, backfill pass for the rest)."""
    out: dict[int, list[float]] = {}
    for c in sorted(data["sam_caches"], key=lambda c: "|backward|" in c["key"]):
        for r in _records((folder / "sam" / c["file"]).read_text()):
            f = r.frame.frame_index if r.frame is not None else 0
            o = obs_by_frame.get(f)
            if o is None or f in out:
                continue
            res = _outline(r)
            if res is None:
                continue
            pts, (mx, my) = res
            # the record must be the tracked object: its outline centre sits on the observation
            if np.hypot(mx - o.cx, my - o.cy) < 0.5 * max(o.right - o.left, o.bottom - o.top):
                out[f] = pts
    return out


# --------------------------------------------------------------------------
# Causal snapshots
# --------------------------------------------------------------------------

def _fits(fits) -> dict:
    return {f.name: {"dbic": None if not f.ok else round(float(f.delta_bic), 2),
                     "params": {k: float(v) for k, v in f.params.items()},
                     "errors": {k: float(v) for k, v in f.errors.items()},
                     "rms": None if not f.ok else float(f.rms)} for f in fits}


def snapshots(exp, run, data: dict, dropped_t: np.ndarray, measured_pivot) -> list[dict]:
    c = run.cycles
    L = pendulum.effective_length(exp.length_m, exp.bob_radius_m)
    end = c.t_mid + c.period / 2                     # when each cycle was complete
    order = np.argsort(end, kind="stable")
    tr = run.track
    u = np.array([np.sin(run.theta_eq), np.cos(run.theta_eq)])
    snaps = []
    for n in range(1, c.period.size + 1):
        idx = np.sort(order[:n])
        t_known = float(end[order[n - 1]])
        sub = pendulum.Cycles(c.t_mid[idx], c.period[idx], c.amplitude[idx], c.crossings, c.peaks_t, c.peaks_a)
        s = {"n": n, "t": t_known, "amp_deg": float(np.degrees(c.amplitude[order[n - 1]])),
             "period_s": float(c.period[order[n - 1]]), "mask_jumps": int((dropped_t <= t_known).sum())}
        if n >= 3:
            gf = pendulum.fit_g(sub, L)
            s["period_laws"] = _fits(gf)
            best = gf[0]
            g, ge = best.params["g"], best.errors["g"]
            s.update(leading=best.name, g=float(g), g_err=float(ge),
                     g_err_total=float(g * np.hypot(ge / g, exp.length_err_m / L)),
                     T0=float(2 * np.pi * np.sqrt(L / g)))
            ex = next(f for f in gf if f.name.startswith("exact"))
            if ex.ok:   # implied length if g were Cairo's, from the exact law's T0
                T0x = 2 * np.pi * np.sqrt(L / ex.params["g"])
                s["implied_L"] = float(G_LOCAL * T0x**2 / (4 * np.pi**2))
                s["T0_exact"] = float(T0x)
            fq = next(f for f in gf if f.name == "free quadratic")
            if fq.ok:
                s["c"], s["c_err"] = float(fq.params["c"]), float(fq.errors["c"])
        if n >= 4:
            s["damping_laws"] = _fits(pendulum.fit_decay(sub))
        # pivot check with the path seen so far (same rule as the measurement)
        if measured_pivot is not None:
            k = tr.t <= t_known
            if k.sum() > 20:
                xc, yc, R, _ = pendulum.fit_circle(tr.x[k], tr.y[k])
                dv = np.array([xc, yc]) - np.array(measured_pivot)
                side = float(abs(dv[0] * u[1] - dv[1] * u[0]))
                s["pivot_side_frac"] = side / R
        snaps.append(s)
    return snaps


# --------------------------------------------------------------------------
# One trial
# --------------------------------------------------------------------------

def build_trial(spec: dict, out: Path, max_mb: float = 20.0) -> dict:
    folder = ROOT / spec["result"]
    data, m, failures = replay(folder)
    run = m.run
    from .experiments import EXPERIMENTS
    exp = EXPERIMENTS[data["experiment"]](**data["params"])
    bad = compare(data["result"]["headline"], m.headline)
    if bad:
        raise SystemExit(f"{spec['result']}: replay does not match the published result: {bad}")

    from .replay import load_observations
    obs_all = load_observations(folder, data)
    obs_in, _ = pendulum.inside_frame(obs_all, tuple(data["frame_size"]))
    by_frame = {o.frame: o for o in obs_in if o.object_id == run.track.object_id}
    tracked = set(int(f) for f in run.track.frames)
    ts = np.array(data["timestamps"], float)
    # mask jumps = frames the track-cleaning rejected (area or position jump), as in the measurement;
    # frames without a camera-motion fix were never offered to it, so they are not counted
    offered = (set(int(f) for f in np.nonzero(run.camera.valid)[0]) if run.camera is not None
               else set(by_frame))
    dropped_t = np.array(sorted(ts[f] for f in by_frame if f in offered and f not in tracked))
    assert len(dropped_t) == run.track.dropped, (len(dropped_t), run.track.dropped)
    outline = outlines(folder, data, by_frame)

    cam = run.camera
    offs = cam.offsets if cam is not None else np.zeros((len(ts), 2))
    mp = data.get("measured_pivot")
    frames = []
    th_by_frame = {int(f): float(np.degrees(t)) for f, t in zip(run.track.frames, run.theta)}
    xy_by_frame = {int(f): (float(x), float(y)) for f, x, y in zip(run.track.frames, run.track.x, run.track.y)}
    for f in range(len(ts)):
        ox, oy = (float(offs[f][0]), float(offs[f][1])) if f < len(offs) else (0.0, 0.0)
        row = {"t": round(float(ts[f]), 5), "fit": [round(run.pivot[0] + ox, 1), round(run.pivot[1] + oy, 1)]}
        if mp is not None:
            row["mp"] = [round(mp[0] + ox, 1), round(mp[1] + oy, 1)]
        if f in xy_by_frame:
            x, y = xy_by_frame[f]
            row.update(b=[round(x + ox, 2), round(y + oy, 2)], th=round(th_by_frame[f], 3))
        if f in outline:
            row["mask"] = outline[f]
        frames.append(row)

    snaps = snapshots(exp, run, data, dropped_t, mp)
    last = snaps[-1]
    for k in ("g", "g_err_total"):   # the page's final numbers are the published ones
        assert abs(last[k] - m.headline[k]) <= 1e-9 * abs(m.headline[k]), (k, last[k], m.headline[k])

    clip_out = out / f"{spec['id']}.mp4"
    _compress(ROOT / spec["clip"], clip_out, max_mb)
    fps = 1 / np.median(np.diff(ts))
    return {
        "id": spec["id"], "title": spec["title"], "short": spec["short"], "setup": spec["setup"],
        "colour": spec["colour"], "result_dir": spec["result"], "clip": clip_out.name,
        "clip_name": Path(spec["clip"]).parent.name + "/" + Path(spec["clip"]).name,
        "frame_size": data["frame_size"], "fps": float(fps), "duration": float(ts[-1]),
        "L": exp.length_m, "L_err": exp.length_err_m, "bob_radius": exp.bob_radius_m,
        "L_eff": pendulum.effective_length(exp.length_m, exp.bob_radius_m),
        "camera_corrected": cam is not None,
        "camera_motion_px": None if cam is None else [round(float(v)) for v in cam.range_px],
        "theta_eq_deg": float(np.degrees(run.theta_eq)),
        "crossings": [round(float(x), 4) for x in run.cycles.crossings],
        "cycle_ends": [round(s["t"], 4) for s in snaps],
        "frames": frames, "snapshots": snaps,
        "published": {"g": m.headline["g"], "g_err_total": m.headline["g_err_total"],
                      "pivot_side_offset_frac": m.headline.get("pivot_side_offset_frac"),
                      "mask_jumps": int(run.track.dropped), "laws": data["result"]["laws"],
                      "summary": data["result"]["summary"], "flags": failures},
        "tracker": data["tracker"], "phrase": data["phrase"],
    }


def _compress(src: Path, dst: Path, max_mb: float) -> None:
    for crf in (26, 29, 32, 35):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-an",
                        "-vf", "scale=-2:'min(960,ih)':out_range=tv", "-c:v", "libx264", "-preset", "slow",
                        "-crf", str(crf), "-pix_fmt", "yuv420p", "-color_range", "tv",
                        "-g", "30", "-keyint_min", "15", "-fps_mode", "passthrough",
                        "-movflags", "+faststart", str(dst)], check=True)
        if dst.stat().st_size <= max_mb * 1e6:
            return
    raise SystemExit(f"{dst} is still over {max_mb} MB")


# --------------------------------------------------------------------------
# Live feed (generated once, every number checked against the data)
# --------------------------------------------------------------------------

def checkpoints(trial: dict) -> list[dict]:
    """Moments worth a comment: first cycle, first fits, status changes, g milestones,
    diagnosis evidence appearing, the end."""
    snaps, out, prev = trial["snapshots"], [], None

    def status(d):
        return None if d is None else "leading" if d <= 2 else "ruled out" if d > 10 else "alive"

    def payload(s, why):
        p = {"why": why, "t": round(s["t"], 2), "cycles_completed": s["n"],
             "last_cycle_period_s": round(s["period_s"], 4), "last_cycle_amplitude_deg": round(s["amp_deg"], 1)}
        if "period_laws" in s:
            p["period_laws"] = {LAW_IDS[k]: {"name": k, "delta_BIC": v["dbic"], "status": status(v["dbic"])}
                                for k, v in s["period_laws"].items()}
            p["leading_law"] = LAW_IDS[s["leading"]]
            p["g"], p["g_sigma"] = round(s["g"], 3), round(s["g_err_total"], 3)
        if "damping_laws" in s:
            p["damping_laws"] = {DAMP_IDS[k]: {"name": k, "delta_BIC": v["dbic"], "status": status(v["dbic"])}
                                 for k, v in s["damping_laws"].items()}
        for k, r in (("implied_L", 3), ("c", 3), ("c_err", 3), ("pivot_side_frac", 3)):
            if k in s:
                p[k] = round(s[k], r)
        p["mask_jump_frames_so_far"] = s["mask_jumps"]
        return p

    for i, s in enumerate(snaps):
        why = []
        if i == 0:
            why.append("first full cycle measured")
        if "period_laws" in s and (prev is None or "period_laws" not in prev):
            why.append("enough cycles to fit the competing period laws")
        if prev and "period_laws" in s and "period_laws" in prev:
            for k, v in s["period_laws"].items():
                a, b = status(prev["period_laws"][k]["dbic"]), status(v["dbic"])
                if a != b:
                    why.append(f"{LAW_IDS[k]} changed from {a} to {b}")
            if s["leading"] != prev["leading"]:
                why.append(f"leading law is now {LAW_IDS[s['leading']]}")
        if prev and "damping_laws" in s and "damping_laws" not in prev:
            why.append("enough cycles to fit the damping laws")
        if "g" in s and abs(s["g"] - G_LOCAL) > 2 * s["g_err_total"] and not (
                prev and "g" in prev and abs(prev["g"] - G_LOCAL) > 2 * prev["g_err_total"]):
            why.append("g now differs from Cairo's 9.79 by more than 2 sigma")
        if "c" in s and abs(s["c"] - 1 / 16) > 2 * s["c_err"] and not (
                prev and "c" in prev and abs(prev["c"] - 1 / 16) > 2 * prev["c_err"]):
            why.append("the free coefficient c now differs from 1/16 by more than 2 sigma")
        if i == len(snaps) - 1:
            why.append("final result")
        if why:
            out.append(payload(s, "; ".join(why)))
        prev = s
    # keep it readable: first, last, and at most 8 in between, spread over the clip
    if len(out) > 10:
        mid = out[1:-1]
        keep = np.linspace(0, len(mid) - 1, 8).round().astype(int)
        out = [out[0]] + [mid[k] for k in sorted(set(keep))] + [out[-1]]
    return out


FEED_PROMPT = """You are Kepler, an autonomous lab, narrating a replay of one real pendulum trial
({title}; L = {L} m measured from pivot to bob centre; local g in Cairo about 9.79 m/s^2).
For each checkpoint below write ONE short message (1-2 sentences) for a live feed, in the first
person plural ("we"). Say what changed and what it means for the hypotheses.
H1 small-angle, H2 series, H3 exact elliptic, H4 free coefficient c. D1 viscous, D2 air drag,
D3 Coulomb, D4 viscous plus air drag. A law is ruled out at delta BIC > 10, leading at <= 2.

STRICT RULES
- Use ONLY numbers that appear in that checkpoint's data, copied exactly as given. No other numbers,
  no rounding to new values, no invented quantities.
- Write math in LaTeX between single dollar signs, e.g. $g = 10.043 \\pm 0.053$, $\\Delta\\mathrm{{BIC}} = 48.8$.
- No markdown, no emoji.
Reply with JSON only: a list of {{"t": <checkpoint t>, "text": "..."}} in the same order.

CHECKPOINTS
{data}"""


def _numbers(text: str) -> list[str]:
    plain = re.sub(r"\\[a-zA-Z]+", " ", text)          # drop LaTeX commands (\frac, \pm, ...)
    return re.findall(r"(?<![A-Za-z_])\d+(?:\.\d+)?", plain)


def _allowed(cp: dict) -> set[str]:
    nums: set[str] = {"1", "2", "3", "4", "10", "16", "9.79", "0"}
    def walk(v):
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, (int, float)):
            x = float(v)
            for d in range(0, 5):
                s = f"{x:.{d}f}"
                nums.add(s)
                nums.add(s.rstrip("0").rstrip(".") if "." in s else s)
            if x != 0 and abs(x) < 1:   # percentages of fractions
                for d in range(0, 3):
                    nums.add(f"{100 * x:.{d}f}")
            nums.add(str(abs(x)) if x < 0 else str(x))
        elif isinstance(v, dict):
            for w in v.values():
                walk(w)
        elif isinstance(v, list):
            for w in v:
                walk(w)
        elif isinstance(v, str):
            for n in re.findall(r"\d+(?:\.\d+)?", v):
                nums.add(n)
    walk(cp)
    return {n.lstrip("-") for n in nums}


def make_feed(trial: dict) -> tuple[list[dict], dict]:
    cps = checkpoints(trial)
    key = hashlib.sha1(json.dumps([cps, config.SCIENTIST_MODEL], sort_keys=True).encode()).hexdigest()[:12]
    prompt = FEED_PROMPT.format(title=trial["title"], L=trial["L"], data=json.dumps(cps, indent=1))
    resp = config.client().chat.completions.create(
        model=config.SCIENTIST_MODEL, max_tokens=32000,
        messages=[{"role": "user", "content": prompt}])
    config.record_spend("scientist", config.chat_cost(config.SCIENTIST_MODEL, getattr(resp, "usage", None)))
    text = resp.choices[0].message.content or ""
    if not text.strip():   # a reasoning model can spend its whole budget thinking
        raise SystemExit(f"feed for {trial['id']}: empty reply (finish_reason={resp.choices[0].finish_reason})")
    msgs = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
    kept, dropped = [], []
    by_t = {round(cp["t"], 2): cp for cp in cps}
    for msg in msgs:
        cp = by_t.get(round(float(msg.get("t", -1)), 2))
        if cp is None:
            dropped.append({"msg": msg, "why": "no matching checkpoint"})
            continue
        bad = [n for n in _numbers(msg["text"]) if n not in _allowed(cp)]
        if bad:
            dropped.append({"msg": msg, "why": f"numbers not in the data: {bad}"})
        else:
            kept.append({"t": cp["t"], "text": msg["text"], "why": cp["why"]})
    meta = {"model": config.SCIENTIST_MODEL, "key": key, "checkpoints": len(cps), "kept": len(kept),
            "dropped": dropped}
    return kept, meta


# --------------------------------------------------------------------------
# Evidence statements
# --------------------------------------------------------------------------

def statements(trials: dict) -> list[dict]:
    """Clickable evidence: the first moment each piece of evidence holds in the causal data."""
    out = []
    k, s = trials["keys"], trials["shoelace"]

    def first(trial, cond):
        return next((x for x in trial["snapshots"] if x["n"] >= MIN_EVIDENCE_CYCLES and cond(x)), None)

    x = first(k, lambda x: "c" in x and x.get("leading") == "free quadratic"
              and abs(x["c"] - 1 / 16) > 2 * x["c_err"])
    if x:
        out.append({"trial": "keys", "t": x["t"], "law": "H4", "diagnosis": "keys", "overlay": "trail",
                    "text": "Keys: the period rises with amplitude faster than the exact law allows"})
    x = first(k, lambda x: x.get("period_laws", {}).get("isochronous (small-angle)", {}).get("dbic") is not None
              and x["period_laws"]["isochronous (small-angle)"]["dbic"] > 10)
    if x:
        out.append({"trial": "keys", "t": x["t"], "law": "H1", "diagnosis": None, "overlay": "angle",
                    "text": "Keys: the small-angle law is ruled out"})
    x = first(k, lambda x: x.get("pivot_side_frac", 0) > 0.02 and x["n"] >= 3)
    if x:
        out.append({"trial": "keys", "t": x["t"], "law": None, "diagnosis": "keys", "overlay": "pivot",
                    "text": "Keys: the swing is not a circle about the measured pivot"})
    x = first(s, lambda x: "g" in x and abs(x["g"] - G_LOCAL) > 2 * x["g_err_total"])
    if x:
        out.append({"trial": "shoelace", "t": x["t"], "law": "H3", "diagnosis": "shoelace", "overlay": "string",
                    "text": "Shoelace: the period implies a pendulum "
                            f"{100 * (s['L'] - s['snapshots'][-1]['implied_L']):.1f} cm shorter than measured"})
    x = first(s, lambda x: x.get("period_laws", {}).get("isochronous (small-angle)", {}).get("dbic") is not None
              and x["period_laws"]["isochronous (small-angle)"]["dbic"] > 10)
    if x:
        out.append({"trial": "shoelace", "t": x["t"], "law": "H1", "diagnosis": None, "overlay": "angle",
                    "text": "Shoelace: the small-angle law is ruled out"})
    x = first(s, lambda x: "damping_laws" in x and x["damping_laws"]["linear (viscous) damping"]["dbic"] is not None
              and x["damping_laws"]["linear (viscous) damping"]["dbic"] > 10)
    if x:
        out.append({"trial": "shoelace", "t": x["t"], "law": "D1", "diagnosis": None, "overlay": "trail",
                    "text": "Shoelace: pure viscous damping is ruled out"})
    return sorted(out, key=lambda e: e["t"])


# --------------------------------------------------------------------------

def build(out: Path, feed: bool = False, reuse: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    trials = {}
    for spec in TRIALS:
        old = out / f"{spec['id']}.json"
        cached = json.loads(old.read_text(encoding="utf-8")) if old.exists() else {}
        if reuse and cached and (out / cached.get("clip", "-")).exists():
            print(f"[showcase] {spec['id']}: reusing {old.name}")
            t = {k: v for k, v in cached.items() if k not in ("feed", "feed_meta")}
        else:
            print(f"[showcase] {spec['id']}: replaying {spec['result']} ...")
            t = build_trial(spec, out)
        key = hashlib.sha1(json.dumps([checkpoints(t), config.SCIENTIST_MODEL],
                                      sort_keys=True).encode()).hexdigest()[:12]
        if cached.get("feed_meta", {}).get("key") == key and cached.get("feed"):   # never reuse an empty feed
            t["feed"], t["feed_meta"] = cached["feed"], cached["feed_meta"]
        elif feed:
            t["feed"], t["feed_meta"] = make_feed(t)
            print(f"[showcase] {spec['id']}: feed {t['feed_meta']['kept']}/{t['feed_meta']['checkpoints']} "
                  f"messages kept, {len(t['feed_meta']['dropped'])} dropped")
        else:
            t["feed"], t["feed_meta"] = [], {"note": "not generated (run with --feed)"}
        trials[spec["id"]] = t
    index = {"trials": [s["id"] for s in TRIALS], "statements": statements(trials), "g_local": G_LOCAL,
             "min_evidence_cycles": MIN_EVIDENCE_CYCLES}
    for tid, t in trials.items():
        (out / f"{tid}.json").write_text(json.dumps(t, separators=(",", ":")), encoding="utf-8")
    (out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    for p in PAGE_SRC.rglob("*"):
        if p.is_file():
            dst = out / p.relative_to(PAGE_SRC)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
    sizes = sorted((p.stat().st_size / 1e6, p.name) for p in out.glob("*") if p.is_file())
    print("[showcase] wrote", out, "·", ", ".join(f"{n} {s:.1f} MB" for s, n in sizes[-6:]))
