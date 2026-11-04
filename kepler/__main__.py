"""Kepler command line.

  python -m kepler synth  out.mp4 --angle 50 --seconds 30
  python -m kepler synth  out.mp4 --experiment cone --n-filters 2 [--drag linear]
  python -m kepler measure clip.mp4 --length 0.800 [--phrase "orange ball"] [--mock orange]
  python -m kepler measure clip.mp4 --experiment cone --n-filters 2 --ref-phrase "green disc" --ref-size 0.15
  python -m kepler lab --length 0.800 --budget 6 [--serial /dev/ttyUSB0] [--camera 0 | --drop ./incoming | --sim]
  python -m kepler lab --experiment cone --budget 5 --sim
  python -m kepler report sessions/<name>
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path


def build_experiment(a, p: argparse.ArgumentParser, sim=None):
    from .experiments import ConeExperiment, PendulumExperiment

    if a.experiment == "pendulum":
        if a.length is None:
            p.error("the pendulum needs --length (pivot to bob centre, metres)")
        return PendulumExperiment(a.length, a.bob_radius, a.length_err,
                                  pivot_phrases=tuple(a.pivot_phrase or ()), pivot_mode=a.pivot,
                                  max_angle_deg=a.max_angle)
    from .calibration import Reference

    phrase, size = a.ref_phrase, a.ref_size
    if sim is not None:   # the simulator's reference disc, unless told otherwise
        phrase, size = phrase or "green disc", size or sim.truth.ref_diameter_m
    if not phrase or not size:
        p.error("the cone needs --ref-phrase and --ref-size (a reference object of known size "
                "in the plane of the fall)")
    return ConeExperiment(Reference(phrase, size, a.ref_kind), ref_size_err_m=a.ref_size_err, g=a.g)


def cmd_synth(a, p):
    if a.experiment == "cone":
        from .synthetic import ConeTruth, render_cone

        t = render_cone(a.out, ConeTruth(n_filters=a.n_filters, drag=a.drag, fps=a.fps or 120.0))
        print(f"wrote {a.out} (truth {t.drag} drag, vt={t.vt:.4f} m/s, g={t.g}, "
              f"reference disc {t.ref_diameter_m} m)")
        return
    from .synthetic import PendulumTruth, render

    t = render(a.out, PendulumTruth(theta0_deg=a.angle, seconds=a.seconds, fps=a.fps or 60.0,
                                    length_m=a.length or 0.8))
    print(f"wrote {a.out} (truth g={t.g}, L={t.length_m} m)")


def cmd_measure(a, p):
    from . import lab

    exp = build_experiment(a, p)
    value = a.n_filters if a.experiment == "cone" else a.angle
    out = Path(a.out or Path(a.clip).with_suffix("")).resolve()
    res, m = lab.measure_trial(a.clip, out, trial=1, experiment=exp, control_value=value,
                               phrase=a.phrase, mock=a.mock, do_qc=not a.no_qc,
                               make_overlay=not a.no_overlay, start=a.start, duration=a.duration)
    print(f"\nusable: {res.usable}  technician: {res.qc_status}  issues: {res.qc_issues}")
    if res.evidence_failures:
        print(f"evidence checks failed: {res.evidence_failures}")
    if m is None:
        return
    print(f"tracked '{res.track_phrase}' ({res.tracker}): {res.summary}")
    for family, fits in res.laws.items():
        print(f"\n{family.capitalize()} (ranked by BIC):")
        for s in fits:
            print("  " + s)
    h = res.headline
    if "g" in h:
        print(f"\ng = {h['g']:.4f} ± {h['g_err_total']:.4f} m/s² "
              f"(fit ± {h['g_err']:.4f}, plus length uncertainty)")
    if "v_terminal" in h:
        print(f"\nterminal velocity = {h['v_terminal']:.4f} ± {h['v_terminal_err_total']:.4f} m/s "
              f"(fit ± {h['v_terminal_err']:.4f}, plus scale calibration)")
        if "drag_exponent" in h:
            print(f"drag exponent n = {h['drag_exponent']:.3f} ± {h['drag_exponent_err']:.3f}")
    for w in res.warnings:
        print("WARNING: " + w)
    print(f"\nplot: {res.plot}\noverlay: {res.overlay}")


def _preflight(exp, a, recorder, *, wait: bool, live: bool) -> None:
    from . import config
    from .hardware import BAR

    print(f"\n{BAR}\n  BEFORE WE START: {exp.title.upper()}, up to {a.budget} trials\n{BAR}")
    for i, line in enumerate(exp.preflight(), 1):
        print(f"  {i}. {line}")
    if getattr(recorder, "folder", None):
        print(f"  Clips go into: {recorder.folder}")
    if live:
        per_s = 2 * 30 * config.SAM_USD_PER_FRAME
        print(f"  Cost: about ${per_s:.3f} per recorded second at 30 fps (${per_s * 60:.2f} per minute); "
              f"the session stops at ${a.max_usd:.2f}.")
    print("  Each trial: the scientist picks an angle, this window tells you how far to pull, you\n"
          "  record and release, then copy the clip. Measuring takes about a minute after each copy.")
    if wait:
        input(">>> Press Enter to start the session ... ")


def cmd_console(a, p):
    from .console import serve

    serve(a.session, port=a.port, open_browser=not a.no_browser)


def cmd_lab(a, p):
    from . import config, hardware
    from .agent import Scientist

    session = Path(a.session or f"sessions/{dt.datetime.now():%Y%m%d-%H%M%S}")
    mock = a.mock
    sim = None
    if a.sim:
        if a.experiment == "pendulum" and a.length is None:
            a.length = 0.8
        from .experiments import EXPERIMENTS

        sim = EXPERIMENTS[a.experiment].simulator()
        if hasattr(sim.truth, "length_m") and a.length is not None:
            sim.truth.length_m = a.length    # rehearse the pendulum you will really use
    exp = build_experiment(a, p, sim=sim)
    hand_run = False
    if sim is not None and a.drop:
        # Rehearsal of a hand-run session: real instructions and drop-folder pickup, simulated physics.
        from .simlab import RehearsalPhone, RehearsalReleaser

        releaser = RehearsalReleaser(sim, exp.manual_instruction, hardware.DropFolderRecorder.start_hint,
                                     auto=not a.interactive)
        recorder = RehearsalPhone(sim, a.drop)
        mock = mock or exp.mock_colour
        hand_run = True
        setting = (f"SIMULATED {exp.title}, not real data (rehearsal of the hand-run drop-folder flow). "
                   f"{exp.truth_note(sim)} ")
    elif sim is not None:
        releaser = recorder = sim
        mock = mock or exp.mock_colour
        setting = f"SIMULATED {exp.title}, not real data. {exp.truth_note(sim)} "
    else:
        if a.serial and a.experiment != "pendulum":
            p.error("the serial releaser only drives the pendulum")
        recorder = hardware.DropFolderRecorder(a.drop) if a.drop else hardware.CameraRecorder(a.camera)
        releaser = (hardware.SerialReleaser(a.serial) if a.serial
                    else hardware.ManualReleaser(exp.manual_instruction, recorder.start_hint))
        hand_run = not a.serial
        setting = "Real apparatus. "
    if hand_run:
        _preflight(exp, a, recorder, wait=sim is None, live=mock is None)
    if mock:
        setting += f"Tracking by the offline colour tracker ({mock}); SAM 3.1 and the technician were not run. "
    else:
        setting += f"Tracking by {config.SAM_MODEL}; technician {config.QC_MODEL}. "
    setting += f"Scientist: {config.SCIENTIST_MODEL}."
    sci = Scientist(session, experiment=exp, budget=a.budget, releaser=releaser, recorder=recorder,
                    mock=mock, reasoning_effort=a.effort, max_usd=a.max_usd, setting=setting)
    sci.run()
    print(f"\nNotebook: {sci.nb.path}")


def cmd_report(a, p):
    from .report import load, write_report

    out = write_report(a.session, a.out)
    nb = load(a.session)
    print(f"wrote {out} ({nb.provenance} data, {len(nb.trials)} trials, {len(nb.findings)} findings)")


def _experiment_args(x: argparse.ArgumentParser) -> None:
    x.add_argument("--experiment", choices=["pendulum", "cone"], default="pendulum")
    g = x.add_argument_group("pendulum")
    g.add_argument("--length", type=float, help="pivot to bob centre, metres")
    g.add_argument("--bob-radius", type=float, default=0.0, help="metres, for a solid sphere")
    g.add_argument("--length-err", type=float, default=0.002, help="tape uncertainty on L, metres")
    g.add_argument("--pivot-phrase", action="append",
                   help="what SAM should track as the pivot (e.g. 'black knot'); repeat to give fallbacks. "
                        "One extra SAM pass per phrase tried")
    g.add_argument("--max-angle", type=float,
                   help="largest release angle the scientist may ask for, degrees (10-75; default 75)")
    g.add_argument("--pivot", choices=["fit", "measured"], default="fit",
                   help="angles about the circle-fit centre (default) or the directly measured pivot; "
                        "the two are always compared as a physics check")
    c = x.add_argument_group("cone")
    c.add_argument("--ref-phrase", help="what SAM segments for scale, e.g. 'green disc'")
    c.add_argument("--ref-size", type=float, help="its known size, metres")
    c.add_argument("--ref-kind", choices=["diameter", "height", "width"], default="diameter")
    c.add_argument("--ref-size-err", type=float, default=0.0005, help="tolerance on the size, metres")
    c.add_argument("--g", type=float, default=9.79, help="local g for the drag laws, m/s²")


def main():
    p = argparse.ArgumentParser(prog="kepler")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("synth", help="render a synthetic clip with known physics")
    s.add_argument("out")
    s.add_argument("--experiment", choices=["pendulum", "cone"], default="pendulum")
    s.add_argument("--angle", type=float, default=50)
    s.add_argument("--seconds", type=float, default=30)
    s.add_argument("--fps", type=float)
    s.add_argument("--length", type=float)
    s.add_argument("--n-filters", type=int, default=1)
    s.add_argument("--drag", choices=["quadratic", "linear"], default="quadratic")
    s.set_defaults(func=cmd_synth)

    m = sub.add_parser("measure", help="measure one clip")
    m.add_argument("clip")
    _experiment_args(m)
    m.add_argument("--angle", type=float, help="pendulum: release angle, only for the technician's context")
    m.add_argument("--n-filters", type=int, default=1, help="cone: filters in the stack")
    m.add_argument("--phrase", help="what SAM should track; default: the technician decides")
    m.add_argument("--mock", help="offline colour tracker instead of SAM (orange/red/green/blue)")
    m.add_argument("--no-qc", action="store_true")
    m.add_argument("--no-overlay", action="store_true")
    m.add_argument("--start", type=float, help="trim: start second (cut the release by hand)")
    m.add_argument("--duration", type=float)
    m.add_argument("--out")
    m.set_defaults(func=cmd_measure)

    l = sub.add_parser("lab", help="autonomous session: the scientist designs and runs the experiments")
    _experiment_args(l)
    l.add_argument("--budget", type=int, default=6)
    l.add_argument("--serial", help="ESP32 port for the servo+magnet releaser (pendulum); default: manual")
    l.add_argument("--camera", default=0, type=lambda v: int(v) if v.isdigit() else v)
    l.add_argument("--drop", help="folder to watch for phone recordings instead of a camera")
    l.add_argument("--sim", action="store_true", help="simulated apparatus (rehearsal, no hardware)")
    l.add_argument("--mock", help="offline colour tracker instead of SAM")
    l.add_argument("--interactive", action="store_true",
                   help="rehearsal (--sim --drop) waits at YOUR TURN for Enter or the console button")
    l.add_argument("--effort", default="high")
    l.add_argument("--max-usd", type=float, default=5.0, help="stop the scientist at this spend")
    l.add_argument("--session")
    l.set_defaults(func=cmd_lab)

    c = sub.add_parser("console", help="live Lab Console in the browser, following a session folder")
    c.add_argument("session")
    c.add_argument("--port", type=int, default=8765)
    c.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    c.set_defaults(func=cmd_console)

    r = sub.add_parser("report", help="HTML report of a session from its NOTEBOOK.md and plots")
    r.add_argument("session")
    r.add_argument("--out", help="default: <session>/report.html")
    r.set_defaults(func=cmd_report)

    a = p.parse_args()
    a.func(a, p)


if __name__ == "__main__":
    main()
