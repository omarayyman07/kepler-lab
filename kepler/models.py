"""Fit competing models to the same data and rank them by BIC.

This is the core of "the agent forms hypotheses and lets data decide".
Each hypothesis is a function y = f(x, *params). We report parameter values
with 1-sigma errors, the residual RMS, BIC, and the BIC gap to the best
model (a gap above ~10 is very strong evidence against a model).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class Model:
    name: str
    func: Callable[..., np.ndarray]
    p0: Sequence[float]
    param_names: Sequence[str]
    bounds: tuple = (-np.inf, np.inf)
    note: str = ""


@dataclass
class Fit:
    name: str
    params: dict[str, float]
    errors: dict[str, float]
    rms: float
    bic: float
    n: int
    k: int
    delta_bic: float = 0.0
    ok: bool = True
    message: str = ""
    residuals: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)

    def summary(self) -> str:
        if not self.ok:
            return f"{self.name}: fit failed ({self.message})"
        ps = ", ".join(f"{k}={v:.6g}±{self.errors[k]:.2g}" for k, v in self.params.items())
        return f"{self.name}: ΔBIC={self.delta_bic:.1f}  rms={self.rms:.3g}  {ps}"


def fit(model: Model, x, y, sigma=None) -> Fit:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n, k = y.size, len(model.p0)
    try:
        popt, pcov = curve_fit(model.func, x, y, p0=list(model.p0), sigma=sigma,
                               absolute_sigma=sigma is not None, bounds=model.bounds,
                               maxfev=20000)
        res = y - model.func(x, *popt)
        rss = float(np.sum(res**2)) if sigma is None else float(np.sum((res / sigma) ** 2))
        if sigma is None:
            bic = n * np.log(max(rss, 1e-300) / n) + k * np.log(n)
        else:
            bic = rss + k * np.log(n)
        errs = np.sqrt(np.clip(np.diag(pcov), 0, np.inf))
        return Fit(model.name, dict(zip(model.param_names, map(float, popt))),
                   dict(zip(model.param_names, map(float, errs))),
                   float(np.sqrt(np.mean(res**2))), float(bic), n, k, residuals=res)
    except Exception as e:  # noqa: BLE001 - a failed hypothesis is a result, not a crash
        return Fit(model.name, {}, {}, float("nan"), float("inf"), n, k, ok=False, message=str(e))


def compare(models: Sequence[Model], x, y, sigma=None) -> list[Fit]:
    fits = [fit(m, x, y, sigma) for m in models]
    best = min((f.bic for f in fits if f.ok), default=np.inf)
    for f in fits:
        f.delta_bic = f.bic - best if f.ok else np.inf
    return sorted(fits, key=lambda f: f.bic)
