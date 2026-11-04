"""The instrument: raw clip in, checked measurements out. Same for every experiment.

    measure_trial(raw_clip, experiment, ...) ->
      prepare (H.264 MP4, true timestamps)
      -> technician QC (usable? which phrase to track?)
      -> SAM 3.1 track -> experiment.measure (data rows, laws, physics checks)
      -> verdict (QC is a cross-check, not a veto) -> plot/overlay -> pooled CSV
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import hardware, sam, video
from .experiments import Experiment, Measurement, TrialContext


@dataclass
class TrialResult:
    trial: int
    clip: str
    usable: bool
    experiment: str = ""
    setting: str = ""            # e.g. "release at 40°", "3 nested filters"
    qc_issues: list[str] = field(default_factory=list)
    qc_notes: str = ""
    qc_status: str = "skipped"   # pass | rejected | overruled | unchecked | skipped
    evidence_failures: list[str] = field(default_factory=list)
    tracker: str = "SAM 3.1"     # what actually produced the track; the notebook must not claim SAM for a mock
    track_phrase: str = ""
    summary: str = ""
    laws: dict[str, list[str]] = field(default_factory=dict)   # law family -> fit summaries, best first
    headline: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    plot: str = ""
    overlay: str = ""

    def for_agent(self) -> dict:
        d = self.__dict__.copy()
        d.pop("clip", None)
        return d


def measure_trial(raw_clip: str | Path, out_dir: str | Path, *, trial: int, experiment: Experiment,
                  control_value: float, phrase: str | None = None, mock: str | None = None,
                  do_qc: bool = True, make_overlay: bool = True, start: float | None = None,
                  duration: float | None = None) -> tuple[TrialResult, Measurement | None]:
    exp = experiment
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_clip = Path(raw_clip)
    clip = video.prepare_clip(raw_clip, out_dir / f"trial{trial:02d}.mp4", start=start, duration=duration)
    info = video.probe(clip)
    timestamps = frame_times(raw_clip, info, trimmed=start is not None or duration is not None)

    res = TrialResult(trial=trial, clip=str(clip), usable=True, experiment=exp.name,
                      setting=exp.describe(control_value))
    qc_text = exp.qc_experiment(control_value)

    # 1. The technician checks the run and names what to track
    # Upload lazily: if QC and SAM are both cached, the clip never leaves the machine.
    file_id = None
    qc_rejected = False
    if mock is None and do_qc:
        from . import qc
        qc_cache = qc.cache_path(clip, qc_text, rules=exp.qc_rules)
        if not qc_cache.exists():
            file_id = sam.upload(clip)
        rep = qc.check_clip(file_id, qc_text, cache=qc_cache, rules=exp.qc_rules)
        res.qc_issues, res.qc_notes = rep.issues, rep.notes
        phrase = phrase or rep.track_phrase
        print(f"[qc] usable={rep.usable} checked={rep.checked} track='{rep.track_phrase}' issues={rep.issues}")
        if not rep.checked:
            res.usable, res.qc_status = False, "unchecked"
            _save(res, out_dir)
            return res, None
        qc_rejected = not rep.usable
        res.qc_status = "rejected" if qc_rejected else "pass"
    phrase = phrase or exp.default_phrase
    res.track_phrase = phrase

    # 2. SAM 3.1 measures
    if mock is not None:
        from . import mock_sam
        res.tracker = f"offline colour tracker ({mock}), not SAM"
        obs = mock_sam.observations_for_video(str(clip), mock)
        ref_colour = getattr(exp, "mock_reference_colour", "green")
        segment_image = lambda img, _phrase: mock_sam.observations_for_image(img, ref_colour)  # noqa: E731
        segment_video = None
    else:
        obs = sam.segment_video(clip, phrase, file_id=file_id)
        segment_image = sam.segment_image

        def segment_video(concept: str) -> list[sam.Observation]:
            """Extra SAM concepts a plugin needs (e.g. the pivot), cached like the main one."""
            out = sam.segment_video(clip, concept, file_id=file_id)
            sam.save_observations(out, out_dir / f"trial{trial:02d}.{concept.replace(' ', '_')}.obs.json")
            return out
    sam.save_observations(obs, out_dir / f"trial{trial:02d}.obs.json")
    if not obs:
        res.usable = False
        res.evidence_failures = ["SAM found no object"]
        _save(res, out_dir)
        return res, None

    # 3. Physics, by the experiment plugin
    ctx = TrialContext(clip=clip, n_frames=info.n_frames, control_value=control_value,
                       segment_image=segment_image, use_cache=mock is None,
                       frame_size=(info.width, info.height), segment_video=segment_video)
    try:
        m = exp.measure(obs, timestamps, ctx)
    except ValueError as e:   # e.g. never released, calibration object missing
        res.usable = False
        res.evidence_failures = [f"measurement failed: {e}"]
        _save(res, out_dir)
        return res, None
    res.summary, res.headline, res.warnings = m.summary, m.headline, list(m.warnings)
    res.laws = {k: [f.summary() for f in v] for k, v in m.laws.items()}

    # 4. QC is a cross-check, not a veto: a run is rejected only when the technician
    #    AND the SAM/physics evidence agree that it is bad.
    res.evidence_failures = tracking_failures(obs, info.n_frames) + m.evidence_failures
    if qc_rejected and res.evidence_failures:
        res.usable = False
        res.warnings.append("Rejected: the technician and the SAM/physics evidence agree "
                            f"({'; '.join(res.evidence_failures)}).")
    elif qc_rejected:
        res.qc_status = "overruled"
        res.warnings.append("Technician flagged this run, but SAM saw one consistent track and the "
                            "physics checks pass, so it is kept: " + "; ".join(res.qc_issues))
    elif res.evidence_failures:
        res.warnings.append("Evidence checks failed (kept, technician passed it): "
                            + "; ".join(res.evidence_failures))

    if m.laws:
        tracker = "offline colour tracker" if mock is not None else "SAM 3.1"
        p = exp.plot(m, out_dir / f"trial{trial:02d}.png", tracker=tracker)
        res.plot = str(p) if p else ""
        if make_overlay:
            o = exp.overlay(clip, m, out_dir / f"trial{trial:02d}.overlay.mp4")
            res.overlay = str(o) if o else ""
    if res.usable and m.rows:   # rejected runs never enter the pooled data
        append_rows(out_dir / exp.data_file, trial, list(exp.columns), m.rows)
    _save(res, out_dir)
    return res, m


def frame_times(raw_clip: Path, info: video.ClipInfo, *, trimmed: bool) -> np.ndarray:
    """Per-frame times, from the most original source available.

    Re-encoding into MP4 can round timestamps onto a coarser timebase (on a 60 fps
    HEVC phone clip it moved frames by up to 8 ms). The prepared clip keeps every
    frame in order, so when nothing was trimmed frame i of it is frame i of the
    original: use the recorder's sidecar if there is one, else the original's own
    timestamps, else the prepared clip's."""
    if trimmed:
        return info.timestamps
    ts = hardware.load_sidecar_timestamps(raw_clip, info.n_frames)
    if ts is not None:
        return ts
    try:
        src = video.probe(raw_clip)
    except Exception:  # noqa: BLE001 - an unreadable source just falls back
        return info.timestamps
    return src.timestamps if src.n_frames == info.n_frames else info.timestamps


def _save(res: TrialResult, out_dir: Path) -> None:
    (out_dir / f"trial{res.trial:02d}.json").write_text(json.dumps(res.__dict__, indent=2, default=str),
                                                        encoding="utf-8")


def tracking_failures(obs: list[sam.Observation], n_frames: int, *, min_coverage: float = 0.8) -> list[str]:
    """Independent of the technician and of the physics: one object, tracked most of the time."""
    fails = []
    frames_per_id: dict[str, set[int]] = {}
    for o in obs:
        frames_per_id.setdefault(o.object_id, set()).add(o.frame)
    persistent = [i for i, f in frames_per_id.items() if len(f) >= 0.05 * n_frames]
    if len(persistent) != 1:
        fails.append(f"SAM tracked {len(persistent)} persistent objects, expected 1")
    seen = max((len(f) for f in frames_per_id.values()), default=0)
    if seen < min_coverage * n_frames:
        fails.append(f"track covers {seen / n_frames:.0%} of frames")
    return fails


def append_rows(path: Path, trial: int, columns: list[str], rows: list[dict]) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["trial", *columns])
        for r in rows:
            w.writerow([trial, *(r[c] for c in columns)])
