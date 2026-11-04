"""Render clips from exact physics, with the truth written beside them.

Pendulum: the nonlinear equation of motion

    theta'' = -(g/L) sin(theta) - gamma*theta' - beta*theta'*|theta'|

Falling cone: the closed-form fall under linear or quadratic drag, next to a
reference disc of known diameter for metres-per-pixel calibration.

Because the truth is known, this is how we prove each measurement chain
(video -> masks -> tracks -> laws -> numbers) is unbiased before trusting it
on real footage.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class PendulumTruth:
    g: float = 9.80665
    length_m: float = 0.80
    theta0_deg: float = 50.0
    gamma: float = 0.02          # linear damping, 1/s
    beta: float = 0.01           # quadratic damping, 1/rad
    fps: float = 60.0
    seconds: float = 30.0
    width: int = 1280
    height: int = 720
    pivot_px: tuple[float, float] = (640.0, 60.0)
    px_per_m: float = 700.0
    bob_radius_px: int = 14
    noise: float = 4.0
    tilt_deg: float = 1.5        # camera roll, to make sure we don't assume vertical
    shake_px: float = 0.0        # handheld camera: smooth random drift with this peak-to-peak size
    # A deliberately bad run: a second orange ball rolls through the bottom of the
    # swing between these fractions of the clip (QC must reject it).
    intruder: tuple[float, float] | None = None


@dataclass
class ConeTruth:
    """A coffee filter (stack of n nested filters) dropped from rest beside a reference disc."""
    g: float = 9.79              # Cairo; the cone plugin's default local g
    drag: str = "quadratic"      # quadratic | linear
    vt1: float = 1.10            # terminal velocity of one filter, m/s
    n_filters: int = 1           # vt = vt1*sqrt(n) (quadratic) or vt1*n (linear): mass at fixed shape
    drop_m: float = 1.50         # release point to floor
    hold_s: float = 0.40         # held still before release
    after_s: float = 0.40        # resting on the floor after landing
    fps: float = 120.0
    width: int = 720
    height: int = 1280
    px_per_m: float = 650.0
    release_px: tuple[float, float] = (430.0, 110.0)   # filter centre at release
    ref_diameter_m: float = 0.15                        # green reference disc, in the fall plane
    ref_px: tuple[float, float] | None = None           # default: beside the middle of the fall
    sway_m: float = 0.01         # gentle sideways wobble (does not affect the fall)
    noise: float = 4.0
    seconds: float | None = None  # default: hold + fall + after

    @property
    def vt(self) -> float:
        return self.vt1 * (np.sqrt(self.n_filters) if self.drag == "quadratic" else self.n_filters)


def cone_fall_time(t: ConeTruth) -> float:
    from .cone import fall_linear, fall_quadratic

    f = fall_quadratic if t.drag == "quadratic" else fall_linear
    tau = np.linspace(0, 20, 200001)
    return float(tau[np.searchsorted(f(tau, t.vt, t.g), t.drop_m)])


def render_cone(out_path: str | Path, truth: ConeTruth | None = None, seed: int = 0) -> ConeTruth:
    from .cone import fall_linear, fall_quadratic

    truth = truth or ConeTruth()
    rng = np.random.default_rng(seed)
    t_fall = cone_fall_time(truth)
    seconds = truth.seconds or truth.hold_s + t_fall + truth.after_s
    times = np.arange(0.0, seconds, 1.0 / truth.fps)
    tau = np.clip(times - truth.hold_s, 0, None)
    f = fall_quadratic if truth.drag == "quadratic" else fall_linear
    y_m = np.minimum(f(tau, truth.vt, truth.g), truth.drop_m)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    W, H, k = truth.width, truth.height, truth.px_per_m
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), truth.fps, (W, H))
    yy, xx = np.mgrid[0:H, 0:W]
    base = (165 + 30 * yy / H).astype(np.float32)
    background = np.dstack([base, base - 6, base - 12])
    # Reference disc with exact per-pixel area coverage: its size is the calibration
    # truth, and cv2's anti-aliased circle comes out about a pixel too wide.
    r_px = truth.ref_diameter_m * k / 2
    rx, ry = truth.ref_px or (130.0, truth.release_px[1] + truth.drop_m * k / 2)
    d = np.hypot(xx + 0.5 - rx, yy + 0.5 - ry)
    cover = np.clip(r_px + 0.5 - d, 0.0, 1.0)[..., None].astype(np.float32)
    background = background * (1 - cover) + np.array([60, 170, 40], np.float32) * cover
    x0, y0 = truth.release_px
    floor_y = int(round(y0 + truth.drop_m * k + 0.02 * k))       # filter half-height above the floor
    shift = 4
    S = 2**shift
    # fluted basket filter, rim up: trapezoid 12 cm across the rim, 6 cm base, 4 cm tall
    rim, base_w, ht = 0.06 * k, 0.03 * k, 0.04 * k
    for tk, ym in zip(times, y_m):
        img = background + rng.normal(0, truth.noise, background.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.rectangle(img, (0, floor_y), (W, H), (95, 95, 100), -1)
        cx = x0 + truth.sway_m * k * np.sin(2 * np.pi * 0.7 * tk)
        cy = y0 + ym * k
        pts = np.array([[cx - rim, cy - ht / 2], [cx + rim, cy - ht / 2],
                        [cx + base_w, cy + ht / 2], [cx - base_w, cy + ht / 2]])
        cv2.fillPoly(img, [np.round(pts * S).astype(np.int32)], (0, 140, 255), cv2.LINE_AA, shift)
        vw.write(img)
    vw.release()
    d = asdict(truth)
    d.update(vt=truth.vt, fall_s=t_fall, seconds=seconds)
    out_path.with_suffix(".truth.json").write_text(json.dumps(d, indent=2))
    return truth


def simulate(t: PendulumTruth) -> tuple[np.ndarray, np.ndarray]:
    def rhs(_, y):
        th, om = y
        return [om, -(t.g / t.length_m) * np.sin(th) - t.gamma * om - t.beta * om * abs(om)]

    times = np.arange(0.0, t.seconds, 1.0 / t.fps)
    sol = solve_ivp(rhs, (0, t.seconds), [np.radians(t.theta0_deg), 0.0],
                    t_eval=times, rtol=1e-10, atol=1e-12, method="DOP853")
    return times, sol.y[0]


def camera_drift(n: int, fps: float, peak_to_peak: float, seed: int = 0) -> np.ndarray:
    """Smooth handheld drift (n, 2) in px: a random walk low-passed over ~1 s."""
    if peak_to_peak <= 0:
        return np.zeros((n, 2))
    rng = np.random.default_rng(seed + 7919)
    walk = np.cumsum(rng.normal(0, 1, (n, 2)), axis=0)
    k = max(int(fps), 1)
    kernel = np.ones(k) / k
    smooth = np.column_stack([np.convolve(walk[:, i], kernel, mode="same") for i in range(2)])
    smooth -= smooth.mean(axis=0)
    return smooth * (peak_to_peak / max(np.ptp(smooth, axis=0).max(), 1e-9))


def render(out_path: str | Path, truth: PendulumTruth | None = None, seed: int = 0) -> PendulumTruth:
    truth = truth or PendulumTruth()
    rng = np.random.default_rng(seed)
    times, theta = simulate(truth)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), truth.fps,
                         (truth.width, truth.height))
    yy, xx = np.mgrid[0:truth.height, 0:truth.width]
    base = (150 + 40 * xx / truth.width).astype(np.float32)
    background = np.dstack([base, base - 8, base - 15])
    tilt = np.radians(truth.tilt_deg)
    r_px = truth.length_m * truth.px_per_m
    shake = camera_drift(len(times), truth.fps, truth.shake_px, seed)
    shift = 4                    # draw with sub-pixel precision
    S = lambda v: int(round(v * 2**shift))  # noqa: E731
    for k, (tk, th) in enumerate(zip(times, theta)):
        px, py = truth.pivot_px[0] + shake[k, 0], truth.pivot_px[1] + shake[k, 1]
        a = th + tilt
        bx, by = px + r_px * np.sin(a), py + r_px * np.cos(a)
        img = background + rng.normal(0, truth.noise, background.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.line(img, (S(px), S(py)), (S(bx), S(by)), (70, 70, 70), 2, cv2.LINE_AA, shift)
        cv2.circle(img, (S(px), S(py)), 6 * 2**shift, (40, 40, 40), -1, cv2.LINE_AA, shift)
        # the pivot's mounting plate: wall-fixed texture, like a nail and knot
        cv2.rectangle(img, (S(px - 14), S(py - 22)), (S(px + 14), S(py - 8)), (90, 60, 40), -1, cv2.LINE_AA, shift)
        cv2.circle(img, (S(bx), S(by)), truth.bob_radius_px * 2**shift, (0, 140, 255), -1, cv2.LINE_AA, shift)
        if truth.intruder is not None:
            f0, f1 = truth.intruder[0] * truth.seconds, truth.intruder[1] * truth.seconds
            if f0 <= tk <= f1:
                ix = -40 + (truth.width + 80) * (tk - f0) / (f1 - f0)
                cv2.circle(img, (int(ix), int(py + r_px)), truth.bob_radius_px, (0, 140, 255), -1, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    out_path.with_suffix(".truth.json").write_text(json.dumps(asdict(truth), indent=2))
    return truth
