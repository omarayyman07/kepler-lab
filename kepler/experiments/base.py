"""What an experiment plugin is.

The lab (prepare -> QC -> track -> verdict) and the scientist loop are the same
for every experiment. A plugin supplies only what differs:

  * a description for the scientist and rules for the technician
  * one controllable parameter and its range (what run_trial changes)
  * the measurement: SAM observations + frame times -> pooled data rows,
    competing laws ranked by BIC, a headline number, physics checks
  * a simulator, so the whole loop can be rehearsed offline
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..models import Fit
from ..sam import Observation

# (BGR still, noun phrase) -> observations; SAM 3.1 live, or the offline mock
SegmentImage = Callable[[np.ndarray, str], list[Observation]]


@dataclass(frozen=True)
class Control:
    """The one knob the scientist turns between trials."""
    name: str            # run_trial argument name, e.g. "release_angle_deg"
    description: str
    lo: float
    hi: float
    integer: bool = False
    label: str = ""      # for prose, e.g. "release angle"

    def clamp(self, value: float) -> float:
        v = min(self.hi, max(self.lo, float(value)))
        return float(round(v)) if self.integer else v


@dataclass
class TrialContext:
    """Everything a measurement may need besides the track."""
    clip: Path
    n_frames: int
    control_value: float
    segment_image: SegmentImage
    use_cache: bool = True   # False for the offline mock (nothing paid to cache)
    frame_size: tuple[int, int] | None = None   # (width, height) of the prepared clip
    # concept -> SAM observations over the whole clip (live, cached); None offline
    segment_video: Callable[[str], list[Observation]] | None = None
    # Replay (no video): what the live run measured on the picture itself, re-used as recorded
    replay: dict | None = None


@dataclass
class Measurement:
    rows: list[dict]                    # pooled data rows (lab adds the "trial" column)
    laws: dict[str, list[Fit]]          # law family -> fits ranked by BIC
    headline: dict[str, float]          # e.g. {"g": .., "g_err": .., "g_err_total": ..}
    summary: str                        # one line for the notebook and the scientist
    warnings: list[str] = field(default_factory=list)
    evidence_failures: list[str] = field(default_factory=list)
    run: object = None                  # the plugin's own analysis object (plots, overlay)
    replay: dict = field(default_factory=dict)   # video-derived inputs a replay needs (camera track, pivot)


class Experiment(ABC):
    name: str
    title: str
    default_phrase: str                 # what SAM tracks if the technician names nothing
    control: Control
    record: Control                     # record_seconds range
    data_file: str                      # pooled CSV in the session folder
    columns: dict[str, str]             # column -> meaning, shown to the scientist
    mock_colour: str = "orange"         # offline tracker colour of the moving object
    qc_rules: str | None = None         # technician reject rules; None = generic pendulum rules

    @abstractmethod
    def apparatus(self, releaser: str) -> str:
        """Paragraph for the scientist: what is on the bench and what is measured."""

    @abstractmethod
    def goals(self) -> str:
        """Numbered goals and the finishing requirements for the scientist."""

    @abstractmethod
    def describe(self, value: float) -> str:
        """Short trial setting for the notebook heading, e.g. 'release at 40°'."""

    @abstractmethod
    def qc_experiment(self, value: float) -> str:
        """One-line description of the run for the technician."""

    @abstractmethod
    def manual_instruction(self, value: float) -> str:
        """What a human assistant must do to set up this trial."""

    @abstractmethod
    def measure(self, obs: list[Observation], timestamps: np.ndarray, ctx: TrialContext) -> Measurement:
        """Turn one clip's observations into data, fitted laws and physics checks."""

    @classmethod
    @abstractmethod
    def simulator(cls, seed: int = 0):
        """Releaser+recorder rendering exact-physics clips (see simlab.py). A classmethod:
        the CLI needs the simulated apparatus (e.g. its reference object) to build the plugin."""

    def truth_note(self, sim) -> str:
        """Notebook setting line for a simulated session (the true parameters)."""
        return ""

    def replay_params(self) -> dict | None:
        """Constructor arguments that rebuild this plugin for `kepler replay`; None = not supported."""
        return None

    def preflight(self) -> list[str]:
        """Checklist shown to the person once, before a hand-run session starts."""
        return []

    def plot(self, m: Measurement, out_png: Path, tracker: str = "SAM 3.1") -> Path | None:
        """`tracker` names what really produced the track; it goes in the figure."""
        return None

    def overlay(self, clip: Path, m: Measurement, out: Path) -> Path | None:
        return None

    def data_description(self) -> str:
        cols = ", ".join(f"{k} ({v})" for k, v in self.columns.items())
        return f"{self.data_file} (columns: trial, {cols})"
