# Kepler: an autonomous desk lab

**A phone camera is the instrument, Muse Wonder is the scientist, and a human is the hands.**

Kepler runs the scientific method on a real physical experiment. It starts from competing
hypotheses, chooses the experiment that best separates them, checks every recording, measures
it with SAM 3.1, fits the competing laws, and decides what to run next. Every decision and
number goes into a lab notebook.

Built for the **Meta Global AI Developer Hackathon** (Meta × Cerebral Valley, Nov 2026).
Track: Indie Developer. Problem statement: Push the Frontier.

---

## Results

All numbers below come from real runs. Simulated and real data are labelled separately.

### Real footage

| Setup | Result | What the evidence showed |
|---|---|---|
| Tomato on a heavy shoelace, L = 0.985 m ([results/tomato1](results/tomato1)) | g = 10.04 ± 0.05 m/s² (does not cover Cairo's ≈ 9.79) | The period implies an effective length of 0.962 m, 2.3 cm short of the measured L. Automatic flag: the swing is not a circle about the measured pivot (centre 5.3% of the radius to the side; the phone was ~14° off square). Most likely cause of the short effective length: the lace's own mass (not yet tested by weighing). |
| Keys on a ring, thick doubled cord, L = 0.58 m ([results/real1](results/real1)) | g = 10.30 ± 0.06 m/s² (best-ranked law; the exact law gives 10.21) | Period short, and rising with amplitude faster than the exact law allows (free coefficient 0.095 ± 0.006 against 1/16). Automatic flags: path 6.9% of the radius off the measured pivot, 26 frames with mask jumps. Diagnosis: not a simple pendulum (cord mass, keys swinging on the ring). |

On the shoelace run the analysis still separated the physics cleanly. The small-angle
(isochronous) law was ruled out by ΔBIC = 49, and the damping law was viscous plus air drag:
pure air drag lost by ΔBIC 14, pure viscous by 111.

The two flagged runs are the point. Kepler did not hand over a bare number: its evidence checks
flagged both runs, the error bars honestly fail to cover the local g, and the flagged data
pointed to apparatus that is not a simple pendulum. Neither setup has been re-run with a thin,
light string yet.

Both results reproduce from the cached SAM output with no video and no API key:
`python -m kepler replay results/tomato1`.

### Simulated pendulum (ground truth known)

| Test | Result |
|---|---|
| Measurement chain, live SAM 3.1 on a synthetic clip | g = 9.80601 against a true 9.80665 (0.0065% off) |
| Autonomous session, Muse Wonder as scientist (offline colour tracker standing in for SAM) | Found the exact period law and viscous + air-drag damping; g = 9.80642 ± 0.025 against a true 9.80665. It stopped by itself after 3 of 4 trials, having met its goals ($0.36) |
| Technician A/B test (one clean and one faulty synthetic clip, 5 runs each per model) | Muse Wonder 10/10 correct verdicts, Muse Spark 1.3 7/10 |

### Status

The full autonomous loop has run end to end on a simulated pendulum. On real footage the
measurement pipeline is validated, and the live console works, but **a complete live
autonomous session on real hardware was not finished before the deadline.**

---

## How it works

```
   ┌──────────────── Muse Wonder (scientist) ────────────────┐
   │ keeps competing hypotheses alive, picks the next trial,  │
   │ writes and runs its own analysis code, logs findings     │
   └───────────────┬──────────────────────────▲──────────────┘
          "release at 60°"                     │ per-cycle periods, amplitudes,
                   ▼                           │ laws ranked by BIC
        human releases the bob          ┌──────┴───────┐
                   ▼                    │  physics fit │
           phone camera records         └──────▲───────┘
                   ▼                           │ sub-pixel track
   Muse Wonder (technician) checks the clip ──► SAM 3.1 tracks the bob
```

1. **Plan.** Muse Wonder lists the laws still alive and chooses the release angle where their
   predictions differ most relative to the measurement noise.
2. **Run.** The console tells the human exactly what to do: the angle, how many cm to pull for
   this string length, and when to release.
3. **Check.** Muse Wonder watches the raw clip and rejects bad runs (hand in frame, contact,
   camera motion). Its verdict is cross-checked against SAM and physics evidence, so neither
   model can veto a run alone.
4. **Measure.** SAM 3.1 tracks the bob in every frame. Masks become sub-pixel centroids; the
   pivot comes from a geometric circle fit and is checked against the top of the string found
   on the picture; zero crossings are timed to a fraction of a frame.
5. **Decide.** Competing period laws (small-angle, series, exact elliptic, free amplitude term)
   and damping laws (viscous, air drag, Coulomb, combined) are fitted and ranked by BIC. Then
   the loop repeats.

## Meta models

| Model | Job | Why this model |
|---|---|---|
| **SAM 3.1** | The instrument: detects, segments and tracks the bob in every frame from a text phrase | Pixel-precise masks with stable object identity; video costs $0.20 per 1,000 frames |
| **Muse Wonder 1.0** | The scientist: plans trials, writes analysis code, logs findings | Long-horizon planning and reliable tool calling |
| **Muse Wonder 1.0** | The technician: watches each raw clip before any number is trusted | Won the technician A/B test 10/10 against Muse Spark 1.3's 7/10 |

---

## Quick start

```bash
git clone https://github.com/omarayyman07/kepler-lab.git
cd kepler-lab
pip install -r requirements.txt
python -m pytest -q tests          # offline ground-truth tests, no API key needed
```

Reproduce the real results from cached SAM output (no API key, no video):

```bash
python -m kepler replay results/tomato1
python -m kepler replay results/real1
```

Measure a clip (needs `MODEL_API_KEY` from dev.meta.ai):

```bash
python -m kepler measure clip.mp4 --length 0.80        # one clip -> g, laws, plot, overlay video
python -m kepler measure clip.mp4 --length 0.80 --mock orange   # offline colour tracker instead of SAM
```

Run an autonomous session with the live console:

```bash
python -m kepler console sessions/my_session           # terminal 1: the lab console in your browser
python -m kepler lab --length 0.80 --max-angle 60 --budget 6 --max-usd 5 \
    --drop clips/session --session sessions/my_session # terminal 2: the session
python -m kepler lab --length 0.80 --budget 4 --sim    # rehearsal on a simulated pendulum
```

Afterwards:

```bash
python -m kepler report sessions/my_session            # self-contained report.html
python scripts/make_demo.py sessions/my_session         # stitched demo video
```

**Filming tips that matter:** use a heavy, compact bob on a thin, light string (not keys on a
ring, not a shoelace); measure L from the pivot to the bob's centre; put the phone on a stand,
square to the wall, in landscape; keep the whole swing in frame.

---

## Engineering notes

- **Never trust timing assumptions.** Phones record at a variable frame rate, so every frame's
  real timestamp is read from the original file. Re-encoding moved some frames by up to 8 ms on
  a 60 fps phone clip, so the re-encoded copy's timestamps are not used.
- **Pay once.** SAM and technician results are cached by clip contents, so re-analysis is free,
  and a published result ships with the raw SAM output it came from. A spend cap refuses to
  measure a clip that would take the session over it.
- **Bugs fixed at the cause.** A 0.04 bias in the fitted drag exponent was traced to 4:2:0 chroma
  subsampling snapping the offline tracker's mask edges to a 2-pixel grid. The tracker was fixed
  and a regression test pins the cause.
- **Corrections must earn their place.** Camera-motion correction is kept only if it makes the
  cycle timing more consistent; on the tomato clip it raised the period scatter from 5.3 to
  9.7 ms (the swinging lace crossed the tracked feature), so it was not used.
- **20 offline tests** cover the parser path (SAM 3.1's documented wire format), ground-truth
  physics, camera shake, edge-cut masks, track gaps, the agent loop, and the reproduction of the
  published real results.

## Repository layout

```
kepler/              pipeline, agent, console, experiment plugins
kepler/experiments/  pendulum and falling-filter plugins
tests/               offline ground-truth tests
scripts/             smoke test, technician A/B test, demo-video builder, result publisher
firmware/            optional servo + electromagnet releaser (ESP32/Arduino)
results/             curated real results: cached SAM output, replay data, plots, overlay clips
```

## Limitations (honest)

- Muse Wonder already knows pendulum physics, so "rediscovering" the pendulum is partly recall.
  Kepler's contribution there is the measurement and the data-driven verdict; unfamiliar
  systems are the stronger test.
- No real run has yet given a g whose error bar covers the local value: both real setups were
  non-ideal pendulums. A re-run with a thin, light string is the next step.
- A human releases the bob. The servo releaser firmware exists but is untested.
- The falling-filter experiment is validated on synthetic data only.
- A full live autonomous session on real footage was not completed before the deadline.

## Related work

- M. Schmidt and H. Lipson, "Distilling Free-Form Natural Laws from Experimental Data,"
  *Science* 324 (2009). Discovered conservation laws from motion-tracked pendulums.
- B. Chen et al., "Automated discovery of fundamental variables hidden in experimental data,"
  *Nature Computational Science* (2022). Found state variables directly from video.
- Recent LLM agents choose experiments to discover physics laws, mostly in simulation.

Kepler's angle: bring that closed loop to a real setup built from household objects, with a
general-purpose segmentation model as the measuring instrument and two models cross-checking
each other.

## License

MIT
