"""Ground-truth tests for the falling-cone plugin: synthetic drops with known drag."""

import json
import types

import numpy as np
import pytest

from kepler import agent, calibration, cone, mock_sam, synthetic, video
from kepler.experiments import ConeExperiment, PendulumExperiment, TrialContext

REF = calibration.Reference("green disc", 0.15, "diameter")


def _clip(d, name, **kw):
    truth = synthetic.ConeTruth(**kw)
    synthetic.render_cone(d / f"{name}.raw.mp4", truth)
    return video.prepare_clip(d / f"{name}.raw.mp4", d / f"{name}.mp4"), truth


def _measure(path, truth, exp=None, soft=True):
    exp = exp or ConeExperiment(REF, g=truth.g)
    info = video.probe(path)
    ctx = TrialContext(clip=path, n_frames=info.n_frames, control_value=truth.n_filters,
                       segment_image=lambda img, _p: mock_sam.observations_for_image(img, "green", soft=soft),
                       use_cache=False)
    return exp.measure(mock_sam.observations_for_video(str(path), soft=soft), info.timestamps, ctx)


@pytest.fixture(scope="module")
def drops(tmp_path_factory):
    d = tmp_path_factory.mktemp("cone")
    return {"quadratic": _clip(d, "q2", n_filters=2, drag="quadratic"),
            "linear": _clip(d, "l2", n_filters=2, drag="linear"),
            "short": _clip(d, "s8", n_filters=8, drag="quadratic", drop_m=0.4)}


def test_calibration_from_reference_disc(drops):
    path, truth = drops["quadratic"]
    cal = calibration.calibrate(path, REF, lambda img, _p: mock_sam.observations_for_image(img, "green"),
                                use_cache=False)
    assert abs(cal.m_per_px * truth.px_per_m - 1) < 0.002           # scale within 0.2%
    assert abs(cal.size_px - truth.ref_diameter_m * truth.px_per_m) < 0.2


def test_fall_laws_closed_forms():
    """The numerical power law reduces to the closed forms at n = 1 and n = 2.
    10 µm: the linear interpolation between RK4 steps (g h²/8), far below a pixel (~1.5 mm)."""
    tau = np.linspace(0, 1.5, 50)
    assert np.max(np.abs(cone.fall_power(tau, 1.3, 2.0, 9.79) - cone.fall_quadratic(tau, 1.3, 9.79))) < 1e-5
    assert np.max(np.abs(cone.fall_power(tau, 1.3, 1.0, 9.79) - cone.fall_linear(tau, 1.3, 9.79))) < 1e-5


def test_quadratic_drag_and_terminal_velocity(drops):
    path, truth = drops["quadratic"]
    m = _measure(path, truth)
    fits = {f.name: f for f in m.laws["drag law"]}
    assert fits["linear drag"].delta_bic - fits["quadratic drag"].delta_bic > 100
    assert fits["no drag (free fall)"].delta_bic > 100
    n, n_err = fits["power-law drag"].params["n"], fits["power-law drag"].errors["n"]
    assert abs(n - 2) < 2 * n_err and n_err < 0.015                          # unbiased, not just "close"
    assert abs(m.headline["v_terminal"] / truth.vt - 1) < 0.005              # vt within 0.5%
    assert abs(m.headline["v_terminal"] - truth.vt) < 2 * m.headline["v_terminal_err_total"]
    assert m.evidence_failures == []
    assert m.rows and set(m.rows[0]) == {"n_filters", "t_s", "y_m"}


def test_hard_colour_masks_bias_the_exponent_under_420_chroma(drops):
    """Regression for the cause found: hard colour-threshold masks on 4:2:0 video snap to the
    2-px chroma grid and pull n low (1.97 at 2.5 sigma); coverage-weighted masks do not."""
    path, truth = drops["quadratic"]
    hard = {f.name: f for f in _measure(path, truth, soft=False).laws["drag law"]}["power-law drag"]
    soft = {f.name: f for f in _measure(path, truth).laws["drag law"]}["power-law drag"]
    assert hard.params["n"] < 1.985
    assert abs(soft.params["n"] - 2) < abs(hard.params["n"] - 2) / 3


def test_linear_drag_is_told_apart(drops):
    path, truth = drops["linear"]
    m = _measure(path, truth)
    fits = {f.name: f for f in m.laws["drag law"]}
    assert fits["quadratic drag"].delta_bic - fits["linear drag"].delta_bic > 100
    assert abs(fits["power-law drag"].params["n"] - 1) < 0.1
    assert abs(m.headline["v_terminal"] / truth.vt - 1) < 0.01


def test_flags_a_drop_too_short_for_terminal_velocity(drops):
    path, truth = drops["short"]
    m = _measure(path, truth)
    assert any("terminal velocity" in f for f in m.evidence_failures)


def test_plugins_define_the_scientists_tools():
    exp = ConeExperiment(REF)
    run_trial = agent.tools(exp)[0]["function"]
    props = run_trial["parameters"]["properties"]
    assert set(run_trial["parameters"]["required"]) == {"n_filters", "record_seconds", "rationale"}
    assert props["n_filters"]["type"] == "integer"
    prompt = agent.system_prompt(exp, budget=5, releaser="human assistant")
    assert "drops.csv" in prompt and "sqrt(N)" in prompt and "green disc" in prompt
    # the pendulum keeps its original tool signature
    pend = agent.tools(PendulumExperiment(0.8))[0]["function"]["parameters"]["required"]
    assert pend == ["release_angle_deg", "record_seconds", "rationale"]


def test_cone_agent_loop_mechanics(tmp_path, monkeypatch):
    """Scripted stand-in for the scientist drives a simulated cone session end to end."""
    from kepler import config
    from kepler.simlab import ConeSimLab

    script = [
        ("run_trial", {"n_filters": 2, "record_seconds": 3, "rationale": "test"}),
        ("run_analysis", {"purpose": "count", "code": "print(open('drops.csv').read().count('\\n'))"}),
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

        def create(self, **kw):
            assert kw["tools"][0]["function"]["name"] == "run_trial"
            name, args = script[self.i]
            self.i += 1
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=Msg(name, args, self.i))])

    monkeypatch.setattr(config, "client", lambda: Fake())
    sim = ConeSimLab()
    sci = agent.Scientist(tmp_path, experiment=ConeExperiment(REF, g=sim.truth.g), budget=2,
                          releaser=sim, recorder=sim, mock="orange", setting="SIMULATED falling paper cone.")
    assert sci.run() == "done"
    nb = (tmp_path / "NOTEBOOK.md").read_text(encoding="utf-8")
    assert "Trial 1: 2 nested filters" in nb and "Drag law fits" in nb and "SIMULATED" in nb
    rows = (tmp_path / "drops.csv").read_text().splitlines()
    assert rows[0] == "trial,n_filters,t_s,y_m" and len(rows) > 50
