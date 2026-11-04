"""Ground-truth tests: synthetic pendulum with known g and damping."""

import json
import types

import numpy as np
import pytest

from kepler import mock_sam, pendulum, sam, synthetic, video


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("kep")
    truth = synthetic.PendulumTruth(seconds=12, fps=60, theta0_deg=55)
    synthetic.render(d / "raw.mp4", truth)
    return video.prepare_clip(d / "raw.mp4", d / "clip.mp4"), truth


def test_wire_format_roundtrip(clip):
    """Real meta_sam_parser path: documented wire grammar -> observations."""
    path, _ = clip
    short = video.prepare_clip(path, path.with_name("short.mp4"), duration=1.0)
    text = mock_sam.wire_text_for_video(str(short))
    obs = sam.observations_from_text(text)
    fast = mock_sam.observations_for_video(str(short), soft=False)   # binary, like the wire format
    assert len(obs) == len(fast) > 50
    for a, b in zip(obs, fast):
        assert a.frame == b.frame
        assert abs(a.cx - b.cx) < 0.05 and abs(a.cy - b.cy) < 0.05


def test_sam_cache_survives_reencode(clip):
    """measure re-prepares trialNN.mp4 every run; the paid SAM cache must still hit."""
    path, _ = clip
    a = video.prepare_clip(path, path.with_name("again.mp4"), duration=1.0)
    key = sam.cache_path(a, "orange ball").name
    video.prepare_clip(path, a, duration=1.0)
    assert sam.cache_path(a, "orange ball").name == key
    assert sam.cache_path(a, "pendulum bob").name != key


def test_sam_backfill_recovers_late_start(clip, monkeypatch):
    """Live SAM 3.1 started tracking at frame 70. A backward box-prompt pass from the
    first detection must recover frames 0..69, merged under the concept track's id."""
    import re

    path, _ = clip
    short = video.prepare_clip(path, path.with_name("backfill.mp4"), duration=2.0)
    wire = mock_sam.wire_text_for_video(str(short))
    calls = []

    async def fake_segment(content, *, video, metadata=None):
        text = content[0]["text"]
        calls.append((text, metadata))
        lines = wire.splitlines(keepends=True)
        if (metadata or {}).get("propagation_direction") == "backward":
            n = int(re.match(r"<(\d+)f>\d+<\|box;", text).group(1))
            keep = [re.sub(r"^(<\d+f>)0", r"\g<1>5", ln) for ln in lines
                    if int(re.match(r"<(\d+)f>", ln).group(1)) <= n]       # a different object id
        else:
            keep = [ln for ln in lines if int(re.match(r"<(\d+)f>", ln).group(1)) >= 70]
        return await sam._observations_from_events(sam.events_from_text("".join(keep)), video=True)

    monkeypatch.setattr(sam, "_segment", fake_segment)
    monkeypatch.setattr(sam, "upload", lambda p: "file-test")
    obs = sam.segment_video(short, "orange ball")
    full = mock_sam.observations_for_video(str(short), soft=False)

    assert [o.frame for o in obs] == [o.frame for o in full]          # 0..119, no gap, no duplicates
    assert {o.object_id for o in obs} == {"0"}
    assert max(abs(a.cx - b.cx) + abs(a.cy - b.cy) for a, b in zip(obs, full)) < 0.1
    assert len(calls) == 2 and calls[0][0] == "orange ball"
    assert re.fullmatch(r"<70f>0<\|box;x1=\d+;y1=\d+;x2=\d+;y2=\d+;w=1280;h=720\|>", calls[1][0])
    assert calls[1][1] == {"propagation_direction": "backward"}

    sam.segment_video(short, "orange ball")                            # both passes now cached
    assert len(calls) == 2


def test_evidence_failures(clip):
    """QC is a cross-check: the SAM/physics evidence alone must tell clean from bad."""
    from kepler import lab
    from kepler.experiments import PendulumExperiment

    path, _ = clip
    info = video.probe(path)
    obs = mock_sam.observations_for_video(str(path))
    run = pendulum.analyse(pendulum.build_track(obs, info.timestamps))
    assert lab.tracking_failures(obs, info.n_frames) == []
    assert PendulumExperiment(0.8)._evidence(run) == []

    # a second persistent object (e.g. a ball rolling through) must be caught
    intruder = [sam.Observation(o.frame, "9", 50.0, 50.0, 400.0, 40, 40, 60, 60, 0.9)
                for o in obs if o.frame % 2 == 0]
    fails = lab.tracking_failures(obs + intruder, info.n_frames)
    assert any("2 persistent objects" in f for f in fails)


def test_g_and_laws(clip):
    path, truth = clip
    info = video.probe(path)
    obs = mock_sam.observations_for_video(str(path))
    run = pendulum.analyse(pendulum.build_track(obs, info.timestamps))
    assert abs(run.radius_px - truth.length_m * truth.px_per_m) < 1.0
    fits = pendulum.fit_g(run.cycles, truth.length_m)
    assert fits[0].name == "exact (elliptic integral)"
    assert abs(fits[0].params["g"] - truth.g) / truth.g < 5e-4
    small = next(f for f in fits if f.name.startswith("isochronous"))
    assert small.delta_bic > 50


def test_agent_loop_mechanics(tmp_path, monkeypatch):
    """Scripted stand-in for Muse Spark to exercise the tool loop end to end."""
    from kepler import agent, config
    from kepler.experiments import PendulumExperiment
    from kepler.simlab import SimLab

    script = [
        ("run_trial", {"release_angle_deg": 40, "record_seconds": 10, "rationale": "test"}),
        ("run_analysis", {"purpose": "count", "code": "print(open('cycles.csv').read().count('\\n'))"}),
        ("log_finding", {"claim": "c", "evidence": "e", "confidence": "tentative"}),
        ("finish", {"report": "done"}),
    ]

    class Msg:
        def __init__(self, name, args, i):
            fn = types.SimpleNamespace(name=name, arguments=json.dumps(args))
            self.tool_calls = [types.SimpleNamespace(id=f"c{i}", function=fn)]
            self.content = None

        def model_dump(self, **_):
            return {"role": "assistant", "content": None}

    class Fake:
        def __init__(self):
            self.i = 0
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

        def create(self, **_):
            name, args = script[self.i]
            self.i += 1
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=Msg(name, args, self.i))])

    monkeypatch.setattr(config, "client", lambda: Fake())
    sim = SimLab(synthetic.PendulumTruth(fps=30))
    sci = agent.Scientist(tmp_path, experiment=PendulumExperiment(0.8), budget=2, releaser=sim,
                          recorder=sim, mock="orange", setting="SIMULATED pendulum, not real data.")
    assert sci.run() == "done"
    nb = (tmp_path / "NOTEBOOK.md").read_text(encoding="utf-8")
    assert "Trial 1" in nb and "Conclusion" in nb and (tmp_path / "trial01.png").exists()
    # never let a mock or simulated run read as a real SAM measurement
    assert "SIMULATED pendulum" in nb
    assert "Tracked (offline colour tracker (orange), not SAM)" in nb and "Tracked (SAM" not in nb

    # the report re-presents the notebook and carries the provenance
    from kepler import report

    parsed = report.load(tmp_path)
    assert parsed.provenance == "simulated" and parsed.mock_tracking
    assert len(parsed.trials) == 1 and parsed.trials[0].fits and parsed.trials[0].plot == "trial01.png"
    assert [f.claim for f in parsed.findings] == ["c"] and parsed.conclusion == "done"
    page = report.render_html(tmp_path, parsed)
    assert "SIMULATED DATA" in page and "offline tracker, not SAM" in page and "data:image/png;base64" in page
    # a session without a Setting line is never presented as real
    assert report.parse_notebook("# Kepler lab notebook: pendulum\n").provenance == "unknown"
