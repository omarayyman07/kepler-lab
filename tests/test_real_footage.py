"""Robustness to real footage: handheld camera drift, and tracks with gaps.

Both failures were seen on the first real clip (sessions/real1): the phone drifted
~90 x 150 px during the recording, and SAM lost the bob at the frame edge during
the first big swings, which merged two cycles into one ~3.3 s "period".
"""

import numpy as np
import pytest

from kepler import mock_sam, pendulum, synthetic, video
from kepler.experiments import PendulumExperiment, TrialContext


@pytest.fixture(scope="module")
def shaky(tmp_path_factory):
    d = tmp_path_factory.mktemp("shaky")
    # the drift (±30 px) keeps the pivot hardware and the bob's lowest point inside the frame
    truth = synthetic.PendulumTruth(seconds=12, fps=60, theta0_deg=40, shake_px=60, pivot_px=(640.0, 85.0))
    synthetic.render(d / "raw.mp4", truth)
    return video.prepare_clip(d / "raw.mp4", d / "clip.mp4"), truth


def _measure(path, truth, **kw):
    info = video.probe(path)
    ctx = TrialContext(clip=path, n_frames=info.n_frames, control_value=None, segment_image=None,
                       use_cache=False, frame_size=(info.width, info.height))
    exp = PendulumExperiment(truth.length_m, **kw)
    return exp.measure(mock_sam.observations_for_video(str(path)), info.timestamps, ctx)


def test_camera_drift_is_removed(shaky):
    path, truth = shaky
    raw = _measure(path, truth, stabilise=False)
    fixed = _measure(path, truth)
    assert raw.run.circle_rms_px > 5                       # drift ruins the fixed-pivot geometry
    assert fixed.run.circle_rms_px < 1.0
    assert fixed.run.camera is not None and max(fixed.run.camera.range_px) > 45
    assert fixed.run.camera.valid.mean() > 0.95
    assert abs(fixed.headline["g"] - truth.g) / truth.g < 1e-3
    assert abs(fixed.headline["g"] - truth.g) < abs(raw.headline["g"] - truth.g)


def test_still_camera_is_left_alone(tmp_path):
    truth = synthetic.PendulumTruth(seconds=6, fps=60, theta0_deg=40)
    synthetic.render(tmp_path / "raw.mp4", truth)
    path = video.prepare_clip(tmp_path / "raw.mp4", tmp_path / "clip.mp4")
    m = _measure(path, truth)
    assert m.run.camera is None                           # under 1 px of motion: data untouched


def test_gap_over_a_crossing_does_not_merge_cycles(shaky):
    path, truth = shaky
    info = video.probe(path)
    obs = mock_sam.observations_for_video(str(path))
    # lose the bob for 0.3 s around a crossing, as SAM did at the fast bottom of the swing
    run0 = pendulum.analyse(pendulum.build_track(obs, info.timestamps))
    tc = run0.cycles.crossings[5]
    holed = [o for o in obs if abs(info.timestamps[o.frame] - tc) > 0.15]
    run = pendulum.analyse(pendulum.build_track(holed, info.timestamps))
    T = run.cycles.period
    assert T.max() < 1.2 * np.median(T)                   # no merged double-length cycle
    assert any("gap inside them" in w for w in run.warnings)


def test_edge_cut_masks_are_dropped():
    from kepler.sam import Observation
    inside = Observation(0, "0", 100, 100, 50, 90, 90, 110, 110, 0.9)
    cut = Observation(1, "0", 3, 100, 30, 0, 90, 8, 110, 0.9)
    kept, n = pendulum.inside_frame([inside, cut], (640, 480))
    assert kept == [inside] and n == 1
