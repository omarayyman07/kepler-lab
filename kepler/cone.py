"""From SAM observations to the drag law of a falling paper cone (coffee filter).

A light cone dropped from rest reaches terminal velocity within a few tenths of
a second. Which drag law it obeys is visible in how it gets there:

    no drag         y = g t^2 / 2
    linear drag     dv/dt = g (1 - v/vt)       y = vt t - (vt^2/g)(1 - exp(-g t/vt))
    quadratic drag  dv/dt = g (1 - (v/vt)^2)   y = (vt^2/g) ln cosh(g t/vt)
    power-law drag  dv/dt = g (1 - (v/vt)^n)   (numerical; the data choose n)

and, across trials, in how vt scales with the number of nested filters N
(mass at fixed shape): vt ~ N for linear drag, vt ~ sqrt(N) for quadratic.

Release time and position are unknown, so every law also fits y0 and t0.
Height is measured downward from the release point, in metres via calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import Fit, Model, compare
from .sam import Observation


# --------------------------------------------------------------------------
# Laws (distance fallen after tau seconds from rest)
# --------------------------------------------------------------------------

def _logcosh(x: np.ndarray) -> np.ndarray:
    x = np.abs(x)
    return x + np.log1p(np.exp(-2 * x)) - np.log(2.0)


def fall_free(tau, g):
    return 0.5 * g * tau**2


def fall_linear(tau, vt, g):
    return vt * tau - vt**2 / g * (1 - np.exp(-g * tau / vt))


def fall_quadratic(tau, vt, g):
    return vt**2 / g * _logcosh(g * tau / vt)


def fall_power(tau, vt, n, g, steps: int = 800):
    """RK4 on dv/dt = g(1 - (v/vt)^n) from rest, interpolated at tau."""
    tau = np.asarray(tau, float)
    tmax = max(float(tau.max()), 1e-6)
    h = tmax / steps
    f = lambda v: g * (1 - (np.maximum(v, 0) / vt) ** n)
    v, y = 0.0, 0.0
    ys = np.empty(steps + 1)
    ys[0] = 0.0
    for i in range(steps):
        k1 = f(v); k2 = f(v + h * k1 / 2); k3 = f(v + h * k2 / 2); k4 = f(v + h * k3)
        y += h * (v + h * (k1 + k2 + k3) / 6)   # the y-row of classic RK4 for (y' = v, v' = f(v))
        v += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        ys[i + 1] = y
    return np.interp(tau, np.linspace(0, tmax, steps + 1), ys)


def fall_models(g: float, t_lo: float, t_hi: float) -> list[Model]:
    """Competing hypotheses for y(t), each with free release height y0 and time t0."""
    tau = lambda t, t0: np.maximum(np.asarray(t) - t0, 0.0)
    b3 = ([-np.inf, t_lo, 0.05], [np.inf, t_hi, 30.0])
    return [
        Model("no drag (free fall)", lambda t, y0, t0, ge: y0 + fall_free(tau(t, t0), ge),
              [0.0, t_lo, g], ["y0", "t0", "g_eff"], ([-np.inf, t_lo, 0.1], [np.inf, t_hi, 30.0]),
              "Fits g freely; drag shows up as g_eff < g and structured residuals."),
        Model("linear drag", lambda t, y0, t0, vt: y0 + fall_linear(tau(t, t0), vt, g),
              [0.0, t_lo, 1.0], ["y0", "t0", "vt"], b3, "Stokes-like: drag proportional to v."),
        Model("quadratic drag", lambda t, y0, t0, vt: y0 + fall_quadratic(tau(t, t0), vt, g),
              [0.0, t_lo, 1.0], ["y0", "t0", "vt"], b3, "Newtonian: drag proportional to v^2."),
        Model("power-law drag", lambda t, y0, t0, vt, n: y0 + fall_power(tau(t, t0), vt, n, g),
              [0.0, t_lo, 1.0, 1.5], ["y0", "t0", "vt", "n"],
              ([-np.inf, t_lo, 0.05, 0.5], [np.inf, t_hi, 30.0, 4.0]),
              "Let the data choose the exponent (theory for a cone at Re~10^3-10^4: n = 2)."),
    ]


# --------------------------------------------------------------------------
# Track
# --------------------------------------------------------------------------

@dataclass
class Fall:
    object_id: str
    frames: np.ndarray
    t: np.ndarray
    x: np.ndarray        # m, horizontal, relative to the first point
    y: np.ndarray        # m, downward, relative to the first point
    area: np.ndarray     # px^2
    dropped: int = 0     # frames rejected for mask area jumps (tumbling, occlusion)
    landed: bool = False


def build_fall(obs: list[Observation], timestamps: np.ndarray, m_per_px: float,
               *, pre_release_s: float = 0.2, onset_m: float = 0.005, land_m: float = 0.003) -> Fall:
    """Main moving object -> (t, x, y) in metres, trimmed to [just before release, just before landing]."""
    if not obs:
        raise ValueError("SAM returned no objects. Try a more concrete concept phrase.")
    ids = sorted({o.object_id for o in obs})

    def score(i):
        ys = np.array([o.cy for o in obs if o.object_id == i])
        return (float(np.ptp(ys)) if ys.size > 1 else 0.0, ys.size)
    oid = max(ids, key=score)
    sel = sorted((o for o in obs if o.object_id == oid and o.frame < len(timestamps)), key=lambda o: o.frame)
    frames = np.array([o.frame for o in sel])
    area = np.array([o.area for o in sel])
    good = (area > 0.6 * np.median(area)) & (area < 1.6 * np.median(area))
    dropped = int((~good).sum())
    sel = [o for o, k in zip(sel, good) if k]
    frames, area = frames[good], area[good]
    t = timestamps[frames]
    x = np.array([o.cx for o in sel]) * m_per_px
    y = np.array([o.cy for o in sel]) * m_per_px
    x, y = x - x[0], y - y[0]

    moving = np.nonzero(y > onset_m)[0]
    if moving.size == 0:
        raise ValueError("The object never moved down; was it released?")
    i0 = int(np.searchsorted(t, t[moving[0]] - pre_release_s))
    ymax = y.max()
    at_floor = np.nonzero(y >= ymax - land_m)[0]
    i1 = int(at_floor[0])
    # landed = it reached its lowest point and then stayed there (rather than leaving the frame)
    landed = bool(i1 < y.size - 3 and np.ptp(y[i1:]) < 4 * land_m)
    s = slice(i0, i1)
    return Fall(oid, frames[s], t[s], x[s], y[s], area[s], dropped, landed)


def fit_fall(fall: Fall, g: float) -> list[Fit]:
    moving = np.nonzero(fall.y > 0.005)[0]
    t_on = fall.t[moving[0]] if moving.size else fall.t[0]
    models = fall_models(g, float(fall.t[0]) - 0.5, float(t_on))
    for m in models:  # start each fit at the observed onset
        m.p0 = [0.0, float(t_on) - 0.05, *m.p0[2:]]
    return compare(models, fall.t, fall.y)


def model_by_name(g: float, fall: Fall) -> dict[str, Model]:
    return {m.name: m for m in fall_models(g, float(fall.t[0]) - 0.5, float(fall.t[-1]))}


def speed_at_end(fit: Fit, g: float, fall: Fall, dt: float = 1e-3) -> float:
    """Model speed at the last fitted point (did the run reach terminal velocity?)."""
    f = model_by_name(g, fall)[fit.name].func
    p = list(fit.params.values())
    te = float(fall.t[-1])
    return float((f(np.array([te]), *p) - f(np.array([te - dt]), *p))[0] / dt)
