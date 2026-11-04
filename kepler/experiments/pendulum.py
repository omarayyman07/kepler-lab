"""Pendulum plugin: period-amplitude law, damping law, and g."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .. import pendulum, render, stabilise
from .base import Control, Experiment, Measurement, TrialContext


class PendulumExperiment(Experiment):
    name = "pendulum"
    title = "pendulum"
    default_phrase = "pendulum bob"
    control = Control("release_angle_deg", "release angle in degrees, 5 to 75", 5.0, 75.0,
                      label="release angle")
    record = Control("record_seconds", "10 to 90", 10.0, 90.0)
    data_file = "cycles.csv"
    columns = {"t_mid_s": "time at the middle of the cycle", "amplitude_rad": "cycle amplitude",
               "period_s": "cycle period"}

    def __init__(self, length_m: float, bob_radius_m: float = 0.0, length_err_m: float = 0.002,
                 min_cycles: int = 4, max_rms_px: float = 3.0, max_dropped: float = 0.02,
                 stabilise: bool = True, pivot_phrases: tuple[str, ...] = (),
                 max_pivot_offset: float = 0.02, pivot_mode: str = "fit", max_angle_deg: float | None = None):
        if max_angle_deg is not None:   # what the camera can see, or a hand can release cleanly
            hi = float(min(max(max_angle_deg, 10.0), 75.0))
            self.control = replace(self.control, hi=hi, description=f"release angle in degrees, 5 to {hi:.0f}")
        self.length_m, self.bob_radius_m, self.length_err_m = length_m, bob_radius_m, length_err_m
        self.min_cycles, self.max_rms_px, self.max_dropped = min_cycles, max_rms_px, max_dropped
        self.stabilise = stabilise
        self.pivot_phrases = tuple(pivot_phrases)     # SAM phrases for the pivot, tried in order
        self.max_pivot_offset = max_pivot_offset       # circle centre vs measured pivot, fraction of R
        self.pivot_mode = pivot_mode                   # angles about the "fit" centre or the "measured" pivot

    # ---- words -------------------------------------------------------------

    def apparatus(self, releaser: str) -> str:
        return (f"Apparatus: a pendulum. The string length from pivot to the centre of the bob is\n"
                f"{self.length_m:.4f} m, measured with a tape (assume ±{self.length_err_m * 1000:.0f} mm). "
                f"A {releaser} releases the bob from\n"
                f"any angle between {self.control.lo:.0f} and {self.control.hi:.0f} degrees. A camera films "
                "each run; a technician model checks\n"
                "each clip and SAM 3.1 tracks the bob, giving per-cycle period and amplitude with\n"
                "sub-millisecond timing. Each trial takes about a minute.")

    def goals(self) -> str:
        return ("Goals, in order:\n"
                "1. Find the law that relates the period to the amplitude.\n"
                "2. Find the law that governs how the amplitude decays.\n"
                "3. Measure the local gravitational acceleration g with an honest uncertainty\n"
                "   (include the length uncertainty).\n\n"
                "Notes:\n"
                "- Do not assume the small-angle formula. It is one hypothesis among several.\n"
                "- A cycle's amplitude decays with time, so amplitude is an independent variable you\n"
                "  control by choosing the release angle and how long you record.\n\n"
                "Finish with: the period law, the damping law, g ± uncertainty, and what you would do\n"
                "next with more time.")

    def describe(self, value: float | None) -> str:
        return f"release at {value:.0f}°" if value is not None else "release angle not recorded"

    def qc_experiment(self, value: float | None) -> str:
        # Without an angle, keep the original wording (it is part of cached QC keys).
        if value is None:
            return "a simple pendulum swinging freely"
        return f"pendulum released from about {value:.0f} degrees"

    def pull_cm(self, angle_deg: float) -> tuple[float, float]:
        """Sideways distance from the rest position, and rise, for this angle (cm)."""
        a = np.radians(abs(angle_deg))
        return 100 * self.length_m * np.sin(a), 100 * self.length_m * (1 - np.cos(a))

    def manual_instruction(self, value: float) -> str:
        side = "RIGHT" if value >= 0 else "LEFT"
        d, h = self.pull_cm(value)
        return (f"Pull the bob {d:.1f} cm to the {side} of where it hangs at rest, measured "
                f"sideways ({abs(value):.0f}°; it rises {h:.1f} cm). String taut, bob still.")

    def preflight(self) -> list[str]:
        d, h = self.pull_cm(self.control.hi)
        return [f"Pendulum: L = {self.length_m * 100:.1f} cm from the pivot to the centre of the bob.",
                "Mark the rest position: let the bob hang still and put a strip of tape on the wall "
                "right behind it, with a ruler or cm marks going sideways from it.",
                "Phone on a tripod (not in your hand), landscape, level, at the height of the middle of "
                "the string, pointing square at the wall. Check: tape cm marks look equally spaced at "
                "both sides of the frame.",
                f"Frame must hold the whole swing with margin: at least {2 * d + 20:.0f} cm wide (both "
                f"sides at {self.control.hi:.0f}°) and {100 * self.length_m + 20:.0f} cm tall (pivot to "
                "below the bob). Step back until it fits.",
                "Tape a tape measure vertically on the wall next to the string, so L can be checked "
                "from the video.",
                "Normal video at 30 fps (not slow-motion, not Cinematic). Nothing else moving in frame.",
                "Copy clips in original quality (cable / AirDrop 'All Photos Data'), not via a chat app."]

    # ---- measurement -------------------------------------------------------

    def measure(self, obs, timestamps, ctx: TrialContext) -> Measurement:
        notes: list[str] = []
        n_edge = 0
        if ctx.frame_size is not None:
            obs, n_edge = pendulum.inside_frame(obs, ctx.frame_size)
            if n_edge:
                notes.append(f"{n_edge} frames dropped: the bob was cut off by the frame edge")
        L_eff = pendulum.effective_length(self.length_m, self.bob_radius_m)
        track = pendulum.build_track(obs, timestamps)
        camera = self._stabilise(obs, track, timestamps, ctx)
        if camera is not None:
            # Correcting for camera motion can also add noise (the swinging string crossing the
            # tracked feature). Keep it only if the cycle-to-cycle timing gets more consistent.
            steady = pendulum.build_track(stabilise.apply(obs, camera), timestamps)
            rms_raw = pendulum.period_scatter(pendulum.analyse(track), L_eff)
            rms_cam = pendulum.period_scatter(pendulum.analyse(steady), L_eff)
            if rms_cam < rms_raw:
                track = steady
                notes.append(f"Camera motion removed ({camera.summary()}); period scatter "
                             f"{1000 * rms_raw:.1f} -> {1000 * rms_cam:.1f} ms.")
            else:
                notes.append(f"Camera-motion correction not used: it raised the period scatter "
                             f"from {1000 * rms_raw:.1f} to {1000 * rms_cam:.1f} ms ({camera.summary()}).")
                camera = None
        track.edge = n_edge

        fit_run = pendulum.analyse(track)            # pivot from the circle fit alone
        pivot = self._measured_pivot(ctx, camera, fit_run, notes)
        use_measured = pivot is not None and self.pivot_mode == "measured"
        run = pendulum.analyse(track, pivot=pivot) if use_measured else fit_run
        if pivot is not None:
            notes.append(f"Angles measured about the {'measured pivot' if use_measured else 'circle-fit centre'}.")
        run.camera = camera
        run.fit_pivot = fit_run.pivot if use_measured else None
        run.measured_pivot = pivot
        c = run.cycles
        m = Measurement(rows=[], laws={}, headline={}, warnings=notes + list(run.warnings), run=run,
                        summary=f"{c.period.size} full cycles")
        if camera is not None:
            rx, ry = camera.range_px
            m.headline.update(camera_motion_x_px=rx, camera_motion_y_px=ry)
        m.evidence_failures = self._evidence(run)
        if pivot is not None:
            # Sideways offset only: the hardware's top may sit above the true rotation point, but
            # the rotation point must lie on the line of the hanging string.
            u = np.array([np.sin(fit_run.theta_eq), np.cos(fit_run.theta_eq)])   # rest direction
            dv = np.array(fit_run.pivot) - np.array(pivot)
            side, along = float(abs(dv[0] * u[1] - dv[1] * u[0])), float(dv @ u)
            m.headline.update(pivot_side_offset_px=side, pivot_along_offset_px=along,
                              pivot_side_offset_frac=side / fit_run.radius_px,
                              circle_radius_px=fit_run.radius_px)
            if side > self.max_pivot_offset * fit_run.radius_px:
                m.evidence_failures.append(
                    f"the circle through the bob's path is centred {side:.0f}px ({side / fit_run.radius_px:.1%} "
                    "of its radius) to the side of the measured pivot: the path is not a circle about the pivot "
                    "(camera not square to the swing plane, or the string bends below the pivot)")
        if c.period.size >= 3:
            gf, df = pendulum.fit_g(c, L_eff), pendulum.fit_decay(c)
            m.laws = {"period law": gf, "damping law": df}
            g, g_err = gf[0].params.get("g", np.nan), gf[0].errors.get("g", np.nan)
            m.headline.update(g=g, g_err=g_err,
                              g_err_total=float(g * np.hypot(g_err / g, self.length_err_m / L_eff)))
            w = pendulum.wobble(run, L_eff, gf[0])
            m.headline.update(wobble_deg=w["theta_z_deg"], wobble_dg=w["dg"])
            if w["theta_z_deg"] > 1:
                m.warnings.append(
                    f"Out-of-plane wobble: the bob's distance from the pivot varies by up to "
                    f"{w['shortfall_px']:.0f}px ({w['shortfall_px'] / run.radius_px:.1%} of the radius). Read as "
                    f"toward/away-from-camera motion, that is at most {w['theta_z_deg']:.1f}° out of plane, which "
                    f"would shift g by {w['dg']:+.4f} m/s² ({w['dg'] / g:+.2%}); upper bound, counted in no error bar.")
            amp = np.degrees(c.amplitude)
            m.summary += f", amplitude {amp.min():.1f}°–{amp.max():.1f}°"
            m.rows = [{"t_mid_s": f"{t:.6f}", "amplitude_rad": f"{a:.8f}", "period_s": f"{p:.8f}"}
                      for t, a, p in zip(c.t_mid, c.amplitude, c.period)]
        return m

    def _stabilise(self, obs, track, timestamps, ctx: TrialContext, min_motion_px: float = 1.0,
                   min_valid: float = 0.5):
        """Camera motion from the pivot feature; None when the camera held still, there is no
        feature, or it was found in too few frames to trust."""
        if not self.stabilise or not Path(ctx.clip).exists():
            return None
        cam = stabilise.track_pivot(ctx.clip, track.x, track.y, track.t, timestamps)
        if cam is None or cam.valid.mean() < min_valid:
            return None
        if not max(cam.range_px) >= min_motion_px:   # also False for NaN
            return None
        return cam

    def _measured_pivot(self, ctx: TrialContext, camera, fit_run, notes: list[str]):
        """The pivot measured directly, in the track's coordinates: SAM on the pivot hardware
        ('black knot', 'nail') if asked, else the top end of the dark string, found offline."""
        if ctx.segment_video is not None:
            p = self._sam_pivot(ctx, camera, fit_run, notes)
            if p is not None:
                return p
        if not self.stabilise or not Path(ctx.clip).exists():   # offline measurement disabled
            return None
        tr = fit_run.track
        p = stabilise.string_top(ctx.clip, tr.x, tr.y, tr.frames, fit_run.theta, fit_run.pivot,
                                 offsets=camera.offsets if camera is not None else None)
        if p is None or np.hypot(p[0] - fit_run.pivot[0], p[1] - fit_run.pivot[1]) > 0.25 * fit_run.radius_px:
            notes.append("Pivot not measured directly (no string top found near the circle-fit centre); "
                         "using the circle fit.")
            return None
        notes.append(f"Pivot measured offline as the top end of the dark string, at {p[0]:.1f},{p[1]:.1f}px "
                     f"(scatter {p[2]:.1f}px over the sampled frames); circle-fit centre "
                     f"{fit_run.pivot[0]:.1f},{fit_run.pivot[1]:.1f}px.")
        return p[0], p[1]

    def _sam_pivot(self, ctx: TrialContext, camera, fit_run, notes: list[str]):
        for phrase in self.pivot_phrases:
            try:
                kobs = ctx.segment_video(phrase)
            except Exception as e:  # noqa: BLE001 - a failed pivot pass must not lose the trial
                notes.append(f"SAM pivot pass '{phrase}' failed: {type(e).__name__}.")
                continue
            if camera is not None:
                kobs = stabilise.apply(kobs, camera)
            if ctx.frame_size is not None:
                kobs, _ = pendulum.inside_frame(kobs, ctx.frame_size)
            p = pendulum.fixed_point(kobs, near=fit_run.pivot, max_dist=0.25 * fit_run.radius_px)
            if p is not None:
                notes.append(f"Pivot measured by SAM ('{phrase}') at {p[0]:.1f},{p[1]:.1f}px "
                             f"(moved {p[2]:.1f}px rms); circle-fit centre {fit_run.pivot[0]:.1f},"
                             f"{fit_run.pivot[1]:.1f}px.")
                return p[0], p[1]
            notes.append(f"SAM found no stationary '{phrase}' near the pivot.")
        return None

    def _evidence(self, run: pendulum.PendulumRun) -> list[str]:
        fails = []
        tr = run.track
        if run.circle_rms_px > self.max_rms_px:
            fails.append(f"bob path deviates {run.circle_rms_px:.1f}px rms from a circle")
        if tr.dropped > self.max_dropped * max(tr.t.size + tr.dropped, 1):
            fails.append(f"{tr.dropped} frames with area or position jumps")
        if run.cycles.period.size < self.min_cycles:
            fails.append(f"only {run.cycles.period.size} full cycles")
        return fails

    def plot(self, m: Measurement, out_png: Path, tracker: str = "SAM 3.1") -> Path | None:
        if not m.laws:
            return None
        L_eff = pendulum.effective_length(self.length_m, self.bob_radius_m)
        return render.plot_run(m.run, m.laws["period law"], m.laws["damping law"], L_eff, out_png,
                               tracker=tracker)

    def overlay(self, clip: Path, m: Measurement, out: Path) -> Path | None:
        h = m.headline
        gtxt = f"g = {h['g']:.3f} +/- {h['g_err_total']:.3f} m/s^2" if h else None
        return render.overlay_video(clip, m.run, out, g_text=gtxt)

    # ---- rehearsal ---------------------------------------------------------

    @classmethod
    def simulator(cls, seed: int = 0):
        from ..simlab import SimLab
        return SimLab(seed=seed)

    def truth_note(self, sim) -> str:
        t = sim.truth
        return (f"SimLab renders exact-physics clips (true g = {t.g} m/s², L = {t.length_m} m, "
                f"linear damping {t.gamma}/s, quadratic damping {t.beta}/rad).")
