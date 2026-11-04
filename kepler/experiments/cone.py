"""Falling paper cone (nested coffee filters): the drag law and terminal velocity."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import cone
from ..calibration import Reference, calibrate
from .base import Control, Experiment, Measurement, TrialContext

QC_RULES = """Reject the run if any of these happen: the filter touches anything before it lands
(landing on the floor at the end is expected and fine), a hand touches it after release,
it flips over or tumbles, it leaves the frame before landing, the reference object is
not visible, the camera moves, or the lighting changes sharply."""


class ConeExperiment(Experiment):
    name = "cone"
    title = "falling paper cone"
    default_phrase = "coffee filter"
    control = Control("n_filters", "number of nested coffee filters dropped together, 1 to 8 "
                      "(same shape, N times the mass)", 1, 8, integer=True, label="number of filters")
    record = Control("record_seconds", "2 to 8", 2.0, 8.0)
    data_file = "drops.csv"
    columns = {"n_filters": "filters in the stack", "t_s": "time since release (fitted)",
               "y_m": "distance fallen, metres"}
    mock_colour = "orange"
    mock_reference_colour = "green"
    qc_rules = QC_RULES

    def __init__(self, reference: Reference, *, ref_size_err_m: float = 0.0005, g: float = 9.79,
                 min_points: int = 30, terminal_fraction: float = 0.95, max_drift: float = 0.25,
                 max_dropped: float = 0.05):
        self.reference, self.ref_size_err_m, self.g = reference, ref_size_err_m, g
        self.min_points, self.terminal_fraction = min_points, terminal_fraction
        self.max_drift, self.max_dropped = max_drift, max_dropped

    # ---- words -------------------------------------------------------------

    def apparatus(self, releaser: str) -> str:
        r = self.reference
        return ("Apparatus: paper coffee filters (fluted basket cones). A stack of N nested filters has\n"
                "the same shape and N times the mass of one. A " + releaser + " holds the stack rim up\n"
                "and drops it from rest from the same height each time. A camera films the fall; a\n"
                f"technician model checks each clip; SAM 3.1 tracks the filter, and a reference object\n"
                f"('{r.phrase}', {r.kind} {r.size_m * 100:.1f} cm) in the plane of the fall sets the scale\n"
                f"in metres. Local g is taken as {self.g} m/s². Each trial takes about a minute.")

    def goals(self) -> str:
        return ("Goals, in order:\n"
                "1. Find the drag law: how the drag force depends on speed (fit the fall y(t) of each\n"
                "   drop; the competing laws differ most during the acceleration phase).\n"
                "2. Find how the terminal velocity scales with the number of filters N (mass at fixed\n"
                "   shape). Linear drag predicts vt ∝ N, quadratic drag predicts vt ∝ sqrt(N).\n"
                "3. Measure the terminal velocity of a single filter with an honest uncertainty\n"
                "   (include the scale calibration uncertainty).\n\n"
                "Notes:\n"
                "- Do not assume a drag law. Free fall, linear, quadratic and a free exponent are all\n"
                "  hypotheses. The two goals test the same physics two independent ways.\n"
                "- A heavy stack may not reach terminal velocity before the floor; the trial result\n"
                "  says so. Choose N where the hypotheses' predictions differ most.\n\n"
                "Finish with: the drag law, the vt(N) scaling, vt for one filter ± uncertainty, and what\n"
                "you would do next with more time.")

    def describe(self, value: float) -> str:
        n = int(value)
        return f"{n} nested filter{'s' if n != 1 else ''}"

    def qc_experiment(self, value: float) -> str:
        return (f"a stack of {int(value)} nested paper coffee filters dropped rim-up from rest, "
                f"falling vertically beside a reference object ({self.reference.phrase})")

    def manual_instruction(self, value: float) -> str:
        n = int(value)
        return (f"Nest {n} filter{'s' if n != 1 else ''} and hold the stack rim up at the release mark, "
                "with the reference object in view. Let go cleanly; do not throw.")

    # ---- measurement -------------------------------------------------------

    def measure(self, obs, timestamps, ctx: TrialContext) -> Measurement:
        cal = calibrate(ctx.clip, self.reference, ctx.segment_image, use_cache=ctx.use_cache)
        fall = cone.build_fall(obs, timestamps, cal.m_per_px)
        n = int(ctx.control_value)
        m = Measurement(rows=[], laws={}, headline={"m_per_px": cal.m_per_px}, run=(fall, cal),
                        summary=f"scale {cal.m_per_px * 1000:.4f} mm/px from '{self.reference.phrase}' "
                                f"({cal.size_px:.1f}px)")
        if fall.t.size < self.min_points:
            m.evidence_failures.append(f"only {fall.t.size} tracked points during the fall")
            return m
        fits = cone.fit_fall(fall, self.g)
        m.laws = {"drag law": fits}
        best = next((f for f in fits if f.ok and "vt" in f.params), None)
        if best is None:
            m.evidence_failures.append("no drag law could be fitted")
            return m
        y0, t0 = best.params["y0"], best.params["t0"]
        vt, vt_err = best.params["vt"], best.errors["vt"]
        # scale error: reference size tolerance and ~0.5px on its measured size
        cal_rel = np.hypot(self.ref_size_err_m / self.reference.size_m, 0.5 / cal.size_px)
        m.headline.update(v_terminal=vt, v_terminal_err=vt_err,
                          v_terminal_err_total=float(vt * np.hypot(vt_err / vt, cal_rel)),
                          calibration_rel_err=float(cal_rel))
        power = next((f for f in fits if f.ok and f.name == "power-law drag"), None)
        if power is not None:
            m.headline.update(drag_exponent=power.params["n"], drag_exponent_err=power.errors["n"])
        fallen = float(fall.y.max() - y0)
        m.summary = (f"{fall.t.size} points over {fallen:.2f} m; vt = {vt:.3f} ± "
                     f"{m.headline['v_terminal_err_total']:.3f} m/s ({best.name}); " + m.summary)
        m.rows = [{"n_filters": n, "t_s": f"{t - t0:.6f}", "y_m": f"{y - y0:.6f}"}
                  for t, y in zip(fall.t, fall.y)]

        v_end = cone.speed_at_end(best, self.g, fall)
        if v_end < self.terminal_fraction * vt:
            m.evidence_failures.append(
                f"reached only {v_end / vt:.0%} of terminal velocity before the floor "
                "(drop from higher or use fewer filters)")
        drift = float(np.ptp(fall.x))
        if drift > self.max_drift * max(fallen, 1e-6):
            m.evidence_failures.append(f"drifted {drift * 100:.0f} cm sideways while falling {fallen:.2f} m")
        if fall.dropped > self.max_dropped * (fall.t.size + fall.dropped):
            m.evidence_failures.append(f"{fall.dropped} frames with mask area jumps (tumbling or occlusion)")
        if not fall.landed:
            m.warnings.append("the filter left the frame instead of landing in view")
        return m

    def plot(self, m: Measurement, out_png: Path, tracker: str = "SAM 3.1") -> Path | None:
        if not m.laws:
            return None
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fall, cal = m.run
        models = cone.model_by_name(self.g, fall)
        fits = [f for f in m.laws["drag law"] if f.ok]
        tt = np.linspace(fall.t[0], fall.t[-1], 400)
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
        ax[0].plot(fall.t, fall.y, "o", ms=2.5, label="measured")
        for f in fits:
            ax[0].plot(tt, models[f.name].func(tt, *f.params.values()), label=f"{f.name} (ΔBIC {f.delta_bic:.0f})")
        ax[0].set(xlabel="time (s)", ylabel="fallen (m)", title=f"Fall ({tracker} track, calibrated)")
        ax[0].legend(fontsize=7)
        for f in fits:
            ax[1].plot(fall.t, f.residuals * 1000, ".", ms=3, label=f.name)
        ax[1].axhline(0, color="k", lw=0.6)
        ax[1].set(xlabel="time (s)", ylabel="residual (mm)", title="Residuals")
        ax[1].legend(fontsize=7)
        v = np.gradient(fall.y, fall.t)
        ax[2].plot(fall.t, v, ".", ms=3, label="measured (finite difference)")
        for f in fits:
            yy = models[f.name].func(tt, *f.params.values())
            ax[2].plot(tt, np.gradient(yy, tt), label=f.name)
        ax[2].set(xlabel="time (s)", ylabel="speed (m/s)", title="Speed")
        ax[2].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_png, dpi=130)
        plt.close(fig)
        return Path(out_png)

    # ---- rehearsal ---------------------------------------------------------

    @classmethod
    def simulator(cls, seed: int = 0):
        from ..simlab import ConeSimLab
        return ConeSimLab(seed=seed)

    def truth_note(self, sim) -> str:
        t = sim.truth
        law = "vt1·√N" if t.drag == "quadratic" else "vt1·N"
        return (f"ConeSimLab renders exact-physics drops ({t.drag} drag, vt = {law}, vt1 = {t.vt1} m/s, "
                f"true g = {t.g} m/s², drop {t.drop_m} m, reference disc {t.ref_diameter_m * 100:.0f} cm).")
