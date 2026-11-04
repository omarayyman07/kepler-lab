"""Experiment plugins. Add one by subclassing `Experiment` and registering it here."""

from __future__ import annotations

from .base import Control, Experiment, Measurement, TrialContext
from .cone import ConeExperiment
from .pendulum import PendulumExperiment

EXPERIMENTS: dict[str, type[Experiment]] = {
    PendulumExperiment.name: PendulumExperiment,
    ConeExperiment.name: ConeExperiment,
}

__all__ = ["EXPERIMENTS", "ConeExperiment", "Control", "Experiment", "Measurement",
           "PendulumExperiment", "TrialContext"]
