# CLAUDE.md — Kepler

Project memory for Claude Code. Read this first in every session.

## What this is

Kepler is an autonomous desk lab, built for the **Meta Global AI Developer Hackathon**
(Meta x Cerebral Valley). The deadline is **Nov 4, 2026, 9:00 PM EST**. The author is a
solo indie developer, so prefer working and verified over clever.

- **Camera:** the instrument.
- **SAM 3.1:** tracks the moving object; its masks become sub-pixel measurements.
- **Muse Wonder 1.0, as technician:** watches each raw clip, flags bad runs, and names
  what SAM should track. Its verdict is a cross-check, not a veto (see `lab.py`).
- **Muse Wonder 1.0, as scientist:** plans the next trial, writes and runs analysis code,
  and logs findings. Falls back to Muse Spark 1.3 if it fails on tool calling or is too
  slow in the loop.
- **Servo + electromagnet, or a human hand:** releases the bob.

The headline demo has Kepler choose its own experiments, rediscover the large-amplitude
pendulum law and the damping law, and measure g with an honest error bar.

### Judging rubric (optimise for it)

| Criterion | Weight | What judges look for |
|---|---|---|
| Impact | 30% | Real-world potential, who benefits, fit with the problem statement |
| Creative use of Meta AI tooling | 25% | Beyond a basic integration |
| Technical depth | 20% | Sound, refined engineering |
| Demo | 25% | Works and holds up, plus a video worth watching |

Avoid anything that reads as a "simple image analyzer", a "dashboard as the main
feature", or an "AI for education chatbot". Those are on the hackathon's banned list.

## Commands

On this machine the venv is `D:\kepler\.venv` (C: is nearly full); ffmpeg comes from winget.

```bash
pip install -r requirements.txt
python scripts/smoke_api.py                                # live check of every API call (~$0.10, cached after)
python scripts/ab_technician.py                            # technician model A/B on known-verdict clips (~$0.40, uncached)
python -m pytest -q tests                                  # offline ground-truth tests, ~2 min, must stay green
python -m kepler synth out.mp4 --angle 50 --seconds 30     # synthetic clip with known physics (.truth.json beside it)
python -m kepler measure clip.mp4 --length 0.80            # one clip -> g, laws, plot, overlay video
python -m kepler measure clip.mp4 --length 0.80 --mock orange   # same, offline colour tracker instead of SAM
python -m kepler lab --length 0.80 --budget 4 --sim        # agent session against a simulated pendulum
python -m kepler lab --length 0.80 --budget 6 --camera 0   # real session, webcam, manual release
python -m kepler lab --length 0.80 --budget 6 --drop ./incoming          # phone clips dropped into a folder
python -m kepler lab --length 0.53 --budget 6 --sim --drop clips/session_rehearsal   # rehearse the hand-run flow (simulated phone)
python -m kepler console sessions/<name>                   # live Lab Console at http://127.0.0.1:8765 (reads the session folder + console.json)
python -m kepler replay results/<name>                     # reproduce a published result from cached SAM output (no key, no video)
python -m kepler showcase [--feed]                         # rebuild docs/showcase (GitHub Pages "Kepler Lab Replay"); --feed may call muse-wonder once per trial
python -m kepler lab --length 0.80 --budget 6 --serial /dev/ttyUSB0      # servo + magnet releaser

# falling paper cone (coffee filters): reference object of known size sets metres/pixel
python -m kepler synth cone.mp4 --experiment cone --n-filters 2 [--drag linear]
python -m kepler measure cone.mp4 --experiment cone --n-filters 2 --ref-phrase "green disc" --ref-size 0.15
python -m kepler lab --experiment cone --budget 5 --sim
```

The API key is read from `MODEL_API_KEY`. Never print it, log it, or commit it.

## Code map

| File | Role |
|---|---|
| `kepler/config.py` | Base URL, model IDs (env-overridable), OpenAI clients |
| `kepler/video.py` | Re-encode to H.264 MP4 faststart keeping every frame; per-frame timestamps via ffprobe |
| `kepler/sam.py` | SAM 3.1 calls, `meta_sam_parser` parsing, mask → centroid/area, backward box-prompt backfill of a late start, raw-output cache keyed by clip content (`*.sam-*.txt`) |
| `kepler/qc.py` | Technician (video understanding → JSON verdict + track phrase); empty reply retried once, then "unchecked"; cached (`*.qc-*.txt`) |
| `kepler/experiments/` | **Plugins.** `base.py`: `Experiment` (description, goals, one `Control` + range, `measure` → `Measurement` with data rows, laws ranked by BIC, headline, physics checks; QC rules; simulator; plot/overlay). `pendulum.py`, `cone.py`; register new ones in `__init__.py` |
| `kepler/pendulum.py` | Pendulum physics: track cleaning, circle-fit pivot, angles, sub-frame zero crossings, per-cycle period/amplitude, period and damping law families |
| `kepler/cone.py` | Falling-cone physics: fall laws (free, linear, quadratic, power-law with free n via RK4), track → metres, trim release-to-landing, fits |
| `kepler/calibration.py` | Metres/pixel from a SAM image segment of a reference object (`Reference(phrase, size_m, kind)`); diameter from mask area; cached (`*.cal-*.json`) |
| `kepler/models.py` | Generic curve fitting + BIC ranking (`Model`, `Fit`, `compare`) |
| `kepler/lab.py` | One trial, any experiment: prepare → QC (experiment's rules) → SAM → `experiment.measure` → verdict → plot/overlay → append usable rows to the experiment's CSV. Rejected only when QC **and** the evidence agree (generic: one persistent object, ≥80% coverage; plus the plugin's physics checks) |
| `kepler/agent.py` | Muse Wonder scientist; system prompt and `run_trial` schema come from the plugin; tools: `run_trial`, `run_analysis`, `log_finding`, `finish` |
| `kepler/notebook.py` | Writes the session's `NOTEBOOK.md` |
| `kepler/status.py`, `kepler/console.py`, `kepler/console.html` | Live step file `console.json` written by the lab; stdlib web server + page that follows a session (read-only, 127.0.0.1) |
| `kepler/report.py` | Parses `NOTEBOOK.md` (shared with `scripts/make_demo.py`) and renders the session's HTML report; adds no numbers of its own |
| `kepler/render.py` | Overlay video + 3-panel plot |
| `kepler/hardware.py` | Releasers (manual, serial) and recorders (camera, drop folder) |
| `kepler/simlab.py` | Simulated releaser/recorder for offline rehearsal (`SimLab` pendulum, `ConeSimLab`) |
| `kepler/synthetic.py` | Ground-truth renderers: exact nonlinear pendulum; cone drop under linear or quadratic drag beside an exact-coverage reference disc |
| `kepler/mock_sam.py` | Offline SAM stand-in; `wire_text_for_video` emits SAM's real wire format, `observations_for_video` is the fast path |
| `firmware/kepler_release/` | ESP32/Arduino: `ARM <deg>` → READY, `DROP` → DROPPED, `HOME` → READY |

Sessions are written to `sessions/<timestamp>/`, which is gitignored.

## Meta Model API facts

If something fails, check the docs: https://dev.meta.ai/docs/models, `/docs/sam/*`,
`/docs/video-understanding`.

### General

- Models, set in `kepler/config.py` and overridable through the environment:

  | Role | Default | Env override |
  |---|---|---|
  | Scientist | `muse-wonder-1.0` (fallback `muse-spark-1.3`) | `KEPLER_SCIENTIST_MODEL` |
  | Technician QC | `muse-wonder-1.0` (A/B winner, see Status) | `KEPLER_QC_MODEL` |
  | Tracking | `sam-3.1` | `KEPLER_SAM_MODEL` |

- Base URL `https://api.meta.ai/v1`, OpenAI-compatible SDK.
- Responses, Chat Completions and Messages protocols are all available.

### SAM 3.1

- Model `sam-3.1`, Responses API, `stream=True`.
- Content is two parts:
  - `input_text`: one short noun phrase, a single concept. No commands, no lists, no
    negation.
  - `input_video` with a `file_id` (Files API, `purpose="user_data"`) or a `video_url`.
- `metadata` values are **strings**: `{"mask_encoding": "one_bit", "include_confidence": "true"}`.
- Output is one line per frame: `<Nf>id<|box;x1;y1;x2;y2;w;h[;c]|><|mask;x=0;y=0[;c];data=H,W,<enc>payload|>`.
  - Masks are box-local.
  - Frames with no match emit no line, so always key by the `<Nf>` index.
  - Object ids are stable but sparse.
  - Parse with `meta_sam_parser` (≥0.1.0).
- Object prompts (instead of a phrase, never both in one request): `input_text` holds
  tokens shaped like output lines, e.g. `<70f>0<|box;x1=308;y1=507;x2=339;y2=537;w=1280;h=720|>`
  (media pixels, x2/y2 inclusive; the object id after `<Nf>` is required).
  `metadata.propagation_direction` = `"both"` (default) | `"forward"` | `"backward"`.
- The concept pass can start late: on synthetic clips it emitted nothing for frames
  0–69. `sam.backfill_start` recovers them with a backward box pass (verified live: 70/70).
- Limits:
  - MP4/MOV, H.264/H.265 only.
  - Up to 15,000 frames per request.
  - At most 16 tracked objects per frame.
  - Video is billed at $0.20 per 1,000 frames, with a 50-frame minimum.

### Muse Spark

- Model `muse-spark-1.3`. It is a reasoning model, so set `max_tokens` /
  `max_output_tokens` to **4000 or more**, or replies can come back empty.
- Video input on the Responses API: `input_file` or `input_video` with a `file_id`.
  MP4 only.
- Tool calling uses Chat Completions with the standard OpenAI `tools` format.
- Recommended pattern: Muse Spark decides *what* to segment, SAM 3.1 does the pixels.

## Status

| Area | Status |
|---|---|
| Measurement pipeline | Done, offline-verified: synthetic g recovered to 0.004%, the correct period and damping laws win by large ΔBIC |
| Agent tool loop | Live with muse-wonder-1.0 on `--sim` (colour tracker, no SAM/QC): `sessions/sim2/` stopped itself after 3/4 trials, 9 steps, $0.36; exact period law and viscous+quadratic damping found, g = 9.80642 ± 0.025 vs truth 9.80665. Earlier `sessions/sim1/` (4 trials, 18 steps, $0.69) also found a small cubic envelope term. ~5–16 s per step. Spending cap `--max-usd` (default 5). Notebook opens with a Setting line (simulated vs real, tracker, models) and never credits SAM for a mock track |
| Full live loop (SAM + QC + scientist, real clips) | Not run yet |
| Live API (SAM, Files, QC, scientist) | Works (`scripts/smoke_api.py`). Live `measure` on a 30 s synthetic clip: g = 9.80601 vs truth 9.80665 (0.0065%), session `sessions/synth30/` |
| SAM first frames | Fixed: backward box pass recovers the late start; live smoke clip 300/300 frames, median 0.22 px vs colour tracker |
| QC reliability | Technician A/B, 5 runs per clip (`sessions/ab_technician/results.json`): muse-wonder-1.0 clean 5/5, bad 5/5, ~$0.028/call; muse-spark-1.3 clean 2/5 (hallucinated second bob), bad 5/5, ~$0.010/call. Wonder is the default; QC is a cross-check, not a veto |
| Real footage | `sessions/real1/` (clips/Video #1.MOV, keys on a doubled cord, handheld phone). Camera drift (91×152 px) now removed by tracking the pivot hardware (`kepler/stabilise.py`); edge-cut masks dropped; cycles spanning track gaps rejected. g = 10.21 ± 0.07 (exact law) / 10.30 ± 0.06 (BIC-best free quadratic): **4–5% high, not explained by wobble (≤ +0.8%) or timing (audio-checked)**. Leading hypothesis: cord mass makes it a physical pendulum (needs cord ≈ 24% of keys' mass); untested |
| Experiment plugins | Done offline: pendulum refactored into a plugin (cached live synth30 replays to the same g = 9.80601); 13 tests green |
| Cone exponent bias (n 0.04 low) | Found and fixed in the offline tracker: hard colour masks on 4:2:0 video snap to the 2-px chroma grid (CRF-independent; 4:4:4 unbiased). `mock_sam` now uses unclipped coverage-weighted centroids/areas: n = 2.002 ± 0.001 over 6 seeds (fit σ 0.009), scale +0.1%. **Open:** SAM's `one_bit` masks are binary too; check live whether SAM edges follow chroma (run SAM on a synthetic cone clip, ~$0.05) |
| Tomato clip (`sessions/tomato1/`) | g ≈ 9.44–9.48 in every analysis variant: L is wrong, not the pipeline. Period gives L_eff ≈ 0.962 m; pixels (wall tape, 14° oblique camera corrected) give 0.96–0.99 m vs stated 0.93. Camera correction is now kept only if it lowers the period scatter (here it raised it 5.3 → 9.7 ms: the shoelace crossed the template). Pivot measured offline by tracing the dark string; SAM found no 'black knot'/'nail'/'black shoelace', 'string' stream reset twice |
| Second experiment: falling cone | Done offline, no API spend. Control = number of nested filters (1–8). Synthetic N=2: quadratic truth vt 1.5556 → 1.551 ± 0.009, linear beats it by ΔBIC 685, free exponent n = 1.96 ± 0.015; linear truth vt 2.200 → 2.191 ± 0.013, n = 0.996. Calibration within 0.3% on the mock. Not yet run live (SAM on a real cone, SAM image calibration, cone QC rules) or in a live agent session |
| Demo assets | Groundwork done on `sessions/sim2`: `python -m kepler report <session>` → self-contained `report.html` (findings first, trial timeline with the scientist's reasoning, ranked fit tables, provenance banner SIMULATED / REAL / not recorded); `python scripts/make_demo.py <session>` → 78 s 1280×720 demo, provenance label on every frame, text quoted from the notebook. sim2 plots regenerated with the correct tracker label (same fits). Judge-facing README and real-data sessions still to do |

## Physics and measurement rules

- **L** is measured from the pivot to the **centre** of the bob. A finite spherical bob
  gets the `0.4 r²/L` correction (`effective_length`). The tape uncertainty on L
  usually dominates the error on g, and the reported error must include it.
- **Time** comes from real per-frame timestamps (ffprobe, or the recorder's
  `.timestamps.json` sidecar). Never assume N/fps. Watch for iPhone slow-mo exports
  that are retimed to 30 fps.
- **Amplitude** is measured from video, never taken from the requested release angle.
- **Reference value:** g in Cairo is ≈ 9.79 m/s². Use it to check results, never to
  tune them.
- **New laws** go in as `Model`s and are ranked by BIC. A model only wins on data.

## Working rules

- **Never fabricate results.** Every number in the README, notebook or demo must come
  from an actual run, with its session folder recorded.
- Keep `pytest` green. Add a test for every bug fixed on real data where feasible.
- **Ask the user before:**
  - spending more than ~$5 of API credit in one go (SAM video costs per frame; long
    high-fps clips add up)
  - deleting files
  - changing the firmware serial protocol
- Cache everything that costs money. Never re-run SAM on a clip that is already cached.
- `run_analysis` executes model-written code in a subprocess. That isolates crashes;
  it is not a security sandbox.
- **When the user is needed** (filming, hardware, measuring L, a decision), stop and
  give a short, exact checklist.
- **After each task, report in 5 lines or fewer:** what changed, how it was verified,
  what's still risky.
- **Commits:** small and clearly described. The repo may be judged.
