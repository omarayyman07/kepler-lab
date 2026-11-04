"""From SAM observations to pendulum physics.

Pipeline:
  observations -> clean track (t, x, y)
  -> pivot by geometric circle fit (no calibration and no "camera is level" assumption)
  -> angle theta(t) about the fitted equilibrium
  -> zero crossings (sub-frame, by local line fit) -> per-cycle period T_k
  -> per-cycle amplitude A_k (parabolic peak refinement)
  -> competing laws for T(A) and for the amplitude decay A(t), ranked by BIC
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ellipk

from .models import Fit, Model, compare
from .sam import Observation


# --------------------------------------------------------------------------
# Track
# --------------------------------------------------------------------------

@dataclass
class Track:
    object_id: str
    frames: np.ndarray
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    area: np.ndarray
    dropped: int = 0
    edge: int = 0        # frames dropped because the mask touched the frame edge (cut off)


def inside_frame(obs: list[Observation], frame_size: tuple[int, int],
                 edge_px: float = 2.0) -> tuple[list[Observation], int]:
    """Drop masks touching the frame edge: a cut-off mask's centroid is pulled inward."""
    W, H = frame_size
    kept = [o for o in obs if o.left > edge_px and o.top > edge_px
            and o.right < W - edge_px and o.bottom < H - edge_px]
    return kept, len(obs) - len(kept)


def build_track(obs: list[Observation], timestamps: np.ndarray, object_id: str | None = None,
                frame_size: tuple[int, int] | None = None, edge_px: float = 2.0) -> Track:
    if not obs:
        raise ValueError("SAM returned no objects. Try a more concrete concept phrase.")
    edge = 0
    if frame_size is not None:
        obs, edge = inside_frame(obs, frame_size, edge_px)
        if not obs:
            raise ValueError("The object touches the frame edge in every frame.")
    ids = sorted({o.object_id for o in obs})
    if object_id is None:
        # The moving object seen in the most frames is the one we measure.
        def score(i):
            pts = np.array([(o.cx, o.cy) for o in obs if o.object_id == i])
            return (len(pts), float(np.var(pts[:, 0]) + np.var(pts[:, 1])) if len(pts) > 1 else 0.0)
        object_id = max(ids, key=score)
    sel = sorted((o for o in obs if o.object_id == object_id), key=lambda o: o.frame)
    frames = np.array([o.frame for o in sel])
    keep = frames < len(timestamps)
    sel = [o for o, k in zip(sel, keep) if k]
    frames = frames[keep]
    x = np.array([o.cx for o in sel])
    y = np.array([o.cy for o in sel])
    area = np.array([o.area for o in sel])

    # Reject frames where the mask is clearly wrong (merged with hand, half
    # occluded): area far from typical, or an impossible jump.
    med = np.median(area)
    good = (area > 0.6 * med) & (area < 1.6 * med)
    if good.sum() > 10:
        step = np.hypot(np.diff(x), np.diff(y))
        lim = 6 * np.median(step) + 2
        jump = np.r_[False, step > lim] | np.r_[step > lim, False]
        good &= ~jump
    dropped = int((~good).sum())
    return Track(object_id, frames[good], timestamps[frames[good]], x[good], y[good], area[good], dropped, edge)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Geometric least-squares circle; returns (xc, yc, R, rms residual px)."""
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x**2 + y**2
    (xc, yc, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    r0 = np.sqrt(c + xc**2 + yc**2)
    res = least_squares(lambda p: np.hypot(x - p[0], y - p[1]) - p[2], [xc, yc, r0], loss="soft_l1", f_scale=1.0)
    xc, yc, r = res.x
    rms = float(np.sqrt(np.mean((np.hypot(x - xc, y - yc) - r) ** 2)))
    return float(xc), float(yc), float(r), rms


# --------------------------------------------------------------------------
# Oscillation analysis
# --------------------------------------------------------------------------

@dataclass
class Cycles:
    t_mid: np.ndarray        # time at the middle of each full cycle
    period: np.ndarray       # s
    amplitude: np.ndarray    # rad, mean of the two half-swing peaks inside the cycle
    crossings: np.ndarray    # s
    peaks_t: np.ndarray
    peaks_a: np.ndarray      # signed rad


@dataclass
class PendulumRun:
    track: Track
    pivot: tuple[float, float]
    radius_px: float
    circle_rms_px: float
    arc_span_deg: float
    theta_eq: float
    theta: np.ndarray        # rad about equilibrium, aligned with track.t
    cycles: Cycles
    warnings: list[str] = field(default_factory=list)
    camera: object = None    # stabilise.CameraMotion when positions were corrected for camera motion
    fit_pivot: tuple[float, float] | None = None   # circle-fit centre, when `pivot` was measured instead
    measured_pivot: tuple[float, float] | None = None   # pivot measured directly (SAM or the string top)


def _equilibrium(theta: np.ndarray) -> float:
    # Damping-safe centre: for each extremum between two opposite extrema,
    # (e_prev + 2 e + e_next)/4 cancels a linear amplitude decay.
    d = np.diff(theta)
    idx = np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0] + 1
    ext = theta[idx]
    if ext.size < 3:
        return float(np.median(theta))
    centres = (ext[:-2] + 2 * ext[1:-1] + ext[2:]) / 4
    return float(np.median(centres))


def _crossing_time(t, phi, i) -> float:
    lo, hi = max(0, i - 1), min(len(t), i + 3)
    a, b = np.polyfit(t[lo:hi] - t[i], phi[lo:hi], 1)
    return float(t[i] - b / a) if a != 0 else float(t[i])


def _peak(t, phi, i0, i1) -> tuple[float, float]:
    seg = np.abs(phi[i0:i1])
    if seg.size == 0:
        return float("nan"), float("nan")
    j = i0 + int(np.argmax(seg))
    lo, hi = max(i0, j - 3), min(i1, j + 4)
    if hi - lo >= 3:
        c = np.polyfit(t[lo:hi] - t[j], phi[lo:hi], 2)
        if c[0] != 0:
            tp = -c[1] / (2 * c[0])
            if abs(tp) <= (t[hi - 1] - t[lo]):
                return float(t[j] + tp), float(np.polyval(c, tp))
    return float(t[j]), float(phi[j])


def analyse(track: Track, pivot: tuple[float, float] | None = None) -> PendulumRun:
    warnings = []
    if pivot is None:
        xc, yc, r, rms = fit_circle(track.x, track.y)
    else:
        xc, yc = pivot
        rr = np.hypot(track.x - xc, track.y - yc)
        r, rms = float(np.median(rr)), float(np.std(rr))
    raw = np.arctan2(track.x - xc, track.y - yc)
    span = float(np.degrees(raw.max() - raw.min()))
    if pivot is None and span < 20:
        warnings.append(f"Arc spans only {span:.0f} degrees, so the pivot fit is weak; pass --pivot.")
    if rms > 3:
        warnings.append(f"Bob path deviates {rms:.1f}px from a circle (string slack or tracking jitter).")
    eq = _equilibrium(raw)
    phi = raw - eq
    t = track.t

    s = np.sign(phi)
    zc = np.where((s[:-1] != 0) & (s[:-1] != s[1:]) & (np.diff(t) < 3 * np.median(np.diff(t))))[0]
    cross = np.array([_crossing_time(t, phi, i) for i in zc])
    # peaks between consecutive crossings
    pk_t, pk_a = [], []
    for a, b in zip(zc[:-1], zc[1:]):
        tp, ap = _peak(t, phi, a + 1, b + 1)
        pk_t.append(tp)
        pk_a.append(ap)
    pk_t, pk_a = np.array(pk_t), np.array(pk_a)

    # A cycle is measured only if the track has no gap inside it: a gap can hide a
    # zero crossing (two cycles would merge into one) or cut off a peak.
    gap_lim = 2.5 * np.median(np.diff(t))
    gap_starts = t[:-1][np.diff(t) > gap_lim]
    gap_ends = t[1:][np.diff(t) > gap_lim]
    periods, amps, mids = [], [], []
    rejected = 0
    for k in range(len(cross) - 2):
        a, b = cross[k], cross[k + 2]
        if np.any((gap_ends > a) & (gap_starts < b)):
            rejected += 1
            continue
        periods.append(b - a)
        amps.append(0.5 * (abs(pk_a[k]) + abs(pk_a[k + 1])))
        mids.append(0.5 * (a + b))
    cycles = Cycles(np.array(mids), np.array(periods), np.array(amps), cross, pk_t, pk_a)
    if rejected:
        warnings.append(f"{rejected} cycle(s) skipped because the track has a gap inside them "
                        "(object lost or cut off by the frame edge).")
    if len(periods) < 4:
        warnings.append("Fewer than 4 full cycles were found; record a longer clip.")
    return PendulumRun(track, (xc, yc), r, rms, span, eq, phi, cycles, warnings)


# --------------------------------------------------------------------------
# Laws
# --------------------------------------------------------------------------

def effective_length(length_m: float, bob_radius_m: float = 0.0) -> float:
    """Pivot-to-centre length, corrected for a solid spherical bob."""
    return length_m + 0.4 * bob_radius_m**2 / length_m


def period_models(L: float) -> list[Model]:
    """Competing hypotheses for T(A) at fixed effective length L."""
    return [
        Model("isochronous (small-angle)", lambda A, g: np.full_like(A, 2 * np.pi * np.sqrt(L / g)),
              [9.8], ["g"], ([1.0], [30.0]),
              "Textbook: period independent of amplitude."),
        Model("quadratic correction", lambda A, g: 2 * np.pi * np.sqrt(L / g) * (1 + A**2 / 16),
              [9.8], ["g"], ([1.0], [30.0]),
              "First-order large-angle series."),
        Model("exact (elliptic integral)", lambda A, g: 4 * np.sqrt(L / g) * ellipk(np.sin(A / 2) ** 2),
              [9.8], ["g"], ([1.0], [30.0]),
              "Exact solution of theta'' = -(g/L) sin(theta)."),
        Model("free quadratic", lambda A, g, c: 2 * np.pi * np.sqrt(L / g) * (1 + c * A**2),
              [9.8, 0.06], ["g", "c"], ([1.0, -1.0], [30.0, 1.0]),
              "Let the data choose the amplitude coefficient (theory: c = 1/16 = 0.0625)."),
    ]


def decay_models() -> list[Model]:
    """Competing hypotheses for amplitude decay A(t)."""
    def both(t, A0, a, b):
        e = np.exp(-a * t)
        return a * A0 * e / (a + b * A0 * (1 - e))
    return [
        Model("linear (viscous) damping", lambda t, A0, a: A0 * np.exp(-a * t),
              [0.5, 0.01], ["A0", "a"], ([0, 0], [np.pi, 10])),
        Model("quadratic (air drag) damping", lambda t, A0, b: A0 / (1 + b * A0 * t),
              [0.5, 0.01], ["A0", "b"], ([0, 0], [np.pi, 10])),
        Model("dry (Coulomb) friction", lambda t, A0, c: A0 - c * t,
              [0.5, 0.001], ["A0", "c"], ([0, 0], [np.pi, 10])),
        Model("viscous + air drag", both,
              [0.5, 0.005, 0.005], ["A0", "a", "b"], ([0, 1e-9, 0], [np.pi, 10, 10])),
    ]


def period_scatter(run: PendulumRun, L_eff: float) -> float:
    """Residual rms (s) of the exact period law: how self-consistent the cycle timing is.
    Used to choose between analyses of the same clip; it does not depend on the value of g."""
    if run.cycles.period.size < 4:
        return float("inf")
    f = next(f for f in fit_g(run.cycles, L_eff) if f.name.startswith("exact"))
    return f.rms if f.ok else float("inf")


def fixed_point(obs: list[Observation], near: tuple[float, float], max_dist: float,
                min_frames: int = 30) -> tuple[float, float, float] | None:
    """A stationary object (nail, knot) near `near`: its median centroid and rms motion (px)."""
    by_id: dict[str, list[Observation]] = {}
    for o in obs:
        by_id.setdefault(o.object_id, []).append(o)
    best = None
    for os_ in by_id.values():
        if len(os_) < min_frames:
            continue
        x = np.array([o.cx for o in os_]); y = np.array([o.cy for o in os_])
        mx, my = float(np.median(x)), float(np.median(y))
        if np.hypot(mx - near[0], my - near[1]) > max_dist:
            continue
        rms = float(np.sqrt(np.mean((x - mx) ** 2 + (y - my) ** 2)))
        key = (len(os_), -rms)
        if best is None or key > best[0]:
            best = (key, (mx, my, rms))
    return best[1] if best else None


def wobble(run: PendulumRun, L_eff: float, best: Fit) -> dict:
    """How far the bob may leave the image plane, and the most that could shift g.

    In the plane of the image, the bob stays at the radius R from the pivot. Motion
    toward or away from the camera shortens its projected distance by R(1 - cos θz).
    We read the deepest dips below R (5th percentile) as that, which is an upper
    bound: centroid jitter and leftover camera roll dip the radius too. The only way
    θz enters g is through the finite-amplitude correction (it adds to the amplitude
    the period depends on), so we refit the winning law with A_total = √(A² + θz²).
    """
    tr = run.track
    r = np.hypot(tr.x - run.pivot[0], tr.y - run.pivot[1]) - run.radius_px
    shortfall = max(0.0, -float(np.percentile(r, 5)))
    theta_z = float(np.arccos(max(-1.0, 1 - shortfall / run.radius_px)))
    out = {"shortfall_px": shortfall, "theta_z_deg": float(np.degrees(theta_z)), "dg": 0.0}
    c = run.cycles
    if not best.ok or c.amplitude.size < 3 or "g" not in best.params:
        return out
    model = next(m for m in period_models(L_eff) if m.name == best.name)
    from .models import fit as fit_one
    f = fit_one(model, np.hypot(c.amplitude, theta_z), c.period)
    if f.ok:
        out["dg"] = f.params["g"] - best.params["g"]
    return out


def fit_g(cycles: Cycles, L_eff: float) -> list[Fit]:
    return compare(period_models(L_eff), cycles.amplitude, cycles.period)


def fit_decay(cycles: Cycles) -> list[Fit]:
    t = cycles.t_mid - cycles.t_mid[0] if cycles.t_mid.size else cycles.t_mid
    return compare(decay_models(), t, cycles.amplitude)
