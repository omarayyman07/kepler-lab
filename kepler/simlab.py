"""A simulated lab for rehearsing the full agent loop without hardware.

The releaser remembers the angle the scientist asked for (with a little human-
like error); the recorder renders a physically exact clip of that run. Combined
with --mock tracking, the whole loop runs offline except for the scientist itself.

`RehearsalReleaser` + `RehearsalPhone` rehearse a hand-run session: the person
sees exactly the instructions a real session prints, and the simulated "phone"
drops its clip into the real drop folder, so the file pickup path is exercised.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .hardware import DropFolderRecorder, ManualReleaser
from .synthetic import ConeTruth, PendulumTruth, render, render_cone


class SimLab:
    def __init__(self, truth: PendulumTruth | None = None, angle_error_deg: float = 1.5, seed: int = 0):
        self.truth = truth or PendulumTruth(fps=30.0)
        self.err = angle_error_deg
        self.rng = np.random.default_rng(seed)
        self.angle = 20.0

    # releaser interface
    def setup(self, angle_deg: float, seconds: float | None = None) -> None:
        self.angle = angle_deg + self.rng.normal(0, self.err)

    def release(self) -> None:
        pass

    # recorder interface
    def record(self, out: str | Path, seconds: float, on_start=None) -> Path:
        if on_start:
            on_start()
        t = PendulumTruth(**{**self.truth.__dict__, "theta0_deg": self.angle, "seconds": seconds})
        render(out, t, seed=int(self.rng.integers(1 << 30)))
        return Path(out)


class ConeSimLab:
    """Drops a stack of n simulated filters. The recording is cut at the requested length,
    so too short a recording (or too heavy a stack) shows up exactly as it would for real."""

    def __init__(self, truth: ConeTruth | None = None, seed: int = 0):
        self.truth = truth or ConeTruth()
        self.rng = np.random.default_rng(seed)
        self.n = 1

    def setup(self, n_filters: float, seconds: float | None = None) -> None:
        self.n = int(n_filters)

    def release(self) -> None:
        pass

    def record(self, out: str | Path, seconds: float, on_start=None) -> Path:
        if on_start:
            on_start()
        t = ConeTruth(**{**self.truth.__dict__, "n_filters": self.n, "seconds": seconds})
        render_cone(out, t, seed=int(self.rng.integers(1 << 30)))
        return Path(out)


class RehearsalReleaser(ManualReleaser):
    """Prints the real hand-run instructions (no waiting), then sets up the simulator."""

    def __init__(self, sim, instruction, start_hint: str, auto: bool = True):
        super().__init__(instruction, start_hint, auto=auto)
        self.sim = sim

    def setup(self, value: float, seconds: float | None = None) -> None:
        super().setup(value, seconds)
        self.sim.setup(value, seconds)


class RehearsalPhone(DropFolderRecorder):
    """A drop-folder recorder whose 'phone' is the simulator: no real-time countdown."""

    def __init__(self, sim, folder: str | Path):
        super().__init__(folder)
        self.sim = sim
        self.takes = 0

    def _countdown(self, seconds: float) -> None:
        self.takes += 1
        print(f"   (rehearsal: the simulated phone records {seconds:.0f} s and drops the clip in the folder)")
        self.sim.record(self.folder / f"rehearsal_take{self.takes:02d}.mp4", seconds)
