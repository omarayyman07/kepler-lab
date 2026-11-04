"""Kepler's scientist: Muse Wonder running the experimental loop.

The model gets four tools and a trial budget:
  run_trial      choose a release angle and record length; the lab runs it,
                 Muse Spark (as technician) checks the clip, SAM 3.1 measures
                 it, and the per-cycle data is pooled into cycles.csv
  run_analysis   write and run Python against the pooled data (numpy/scipy)
  log_finding    commit a claim with its evidence to the notebook
  finish         final report

What makes it a scientist rather than a script: the textbook formula is
treated as one hypothesis among several, and each trial is chosen to
separate the hypotheses that still survive.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from . import config, lab, status
from .experiments import Experiment
from .notebook import Notebook

SYSTEM = """You are Kepler, an autonomous experimental physicist running a real desk lab.

{apparatus} Budget: {budget} trials.

{goals}

How to work:
- Before each trial, state which hypotheses are still alive and pick the {control}
  and duration where their predictions differ most compared with the measurement noise.
  Put that reasoning in the trial's rationale; it goes in the lab notebook.
- After each trial, analyse the pooled data in {data} with run_analysis. Fit competing
  models and compare them with BIC. Look at residuals.
- If a trial is rejected or looks wrong, say why and decide whether to repeat it.
- Log a finding only when the data supports it, with the numbers. When rounding an
  uncertainty, round it up, never down (0.0245 becomes 0.025, not 0.02).
- Stop early if the goals are met with confidence; do not spend trials for nothing.
"""


def system_prompt(exp: Experiment, budget: int, releaser: str) -> str:
    return SYSTEM.format(apparatus=exp.apparatus(releaser), budget=budget, goals=exp.goals(),
                         control=exp.control.label or exp.control.name, data=exp.data_description())


def tools(exp: Experiment) -> list[dict]:
    c, r = exp.control, exp.record
    return [
        {"type": "function", "function": {
            "name": "run_trial",
            "description": f"Run one {exp.title} trial and record it. Returns the technician's check "
                           f"and this trial's measurement; data is pooled into {exp.data_file}.",
            "parameters": {"type": "object", "properties": {
                c.name: {"type": "integer" if c.integer else "number", "description": c.description},
                r.name: {"type": "number", "description": r.description},
                "rationale": {"type": "string",
                              "description": f"Which hypotheses this trial separates and why this {c.name}"},
            }, "required": [c.name, r.name, "rationale"]}}},
        *TOOLS,
    ]


TOOLS = [
    {"type": "function", "function": {
        "name": "run_analysis",
        "description": "Run Python 3 in the session folder, where the pooled data CSV lives. "
                       "numpy, scipy and matplotlib are available (not pandas). Print what you need "
                       "to see; save any figure as a .png in the current folder.",
        "parameters": {"type": "object", "properties": {
            "purpose": {"type": "string"},
            "code": {"type": "string"},
        }, "required": ["purpose", "code"]}}},
    {"type": "function", "function": {
        "name": "log_finding",
        "description": "Commit a supported claim to the lab notebook.",
        "parameters": {"type": "object", "properties": {
            "claim": {"type": "string"},
            "evidence": {"type": "string", "description": "Numbers: fits, ΔBIC, residuals"},
            "confidence": {"type": "string", "enum": ["tentative", "strong", "conclusive"]},
        }, "required": ["claim", "evidence", "confidence"]}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "End the session with the final report.",
        "parameters": {"type": "object", "properties": {
            "report": {"type": "string"},
        }, "required": ["report"]}}},
]


def run_python(code: str, cwd: Path, timeout: int = 120) -> str:
    """Run model-written analysis code in a separate process.

    This is isolation from crashes and hangs, not a security sandbox: only run
    Kepler on a machine where model-written analysis code is acceptable.
    """
    script = cwd / "_analysis.py"
    script.write_text("import matplotlib\nmatplotlib.use('Agg')\n" + textwrap.dedent(code), encoding="utf-8")
    try:
        p = subprocess.run([sys.executable, script.name], cwd=cwd, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=timeout,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        out = p.stdout + (f"\n[stderr]\n{p.stderr}" if p.returncode else "")
    except subprocess.TimeoutExpired:
        out = f"[timed out after {timeout}s]"
    return out[-6000:] or "[no output]"


class Scientist:
    def __init__(self, session_dir: str | Path, *, experiment: Experiment, budget: int, releaser, recorder,
                 mock: str | None = None, max_steps: int = 60, reasoning_effort: str = "high",
                 max_usd: float = 5.0, setting: str | None = None):
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.exp = experiment
        self.budget, self.trials = budget, 0
        self.releaser, self.recorder, self.mock = releaser, recorder, mock
        self.max_steps, self.effort = max_steps, reasoning_effort
        self.max_usd = max_usd
        self._spend0 = config.total_spend()
        self.nb = Notebook(self.dir, f"Kepler lab notebook: {experiment.title}", setting=setting)
        kind = "servo arm with an electromagnet" if type(releaser).__name__ == "SerialReleaser" else "human assistant"
        self.tools = tools(experiment)
        system = system_prompt(experiment, budget, kind)
        if mock is None:
            system += (f"\n\nMeasuring costs money. SAM 3.1 bills per video frame: about "
                       f"${2 * 30 * config.SAM_USD_PER_FRAME:.3f} per recorded second at 30 fps (a forward "
                       f"and a backfill pass). The whole session is capped at ${max_usd:.2f}; every "
                       "run_trial result reports usd_spent. Record long enough to answer the question, "
                       "not longer.")
        self.messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "The lab is ready. Plan your first trial."},
        ]

    # ---- tools -------------------------------------------------------------

    def run_trial(self, rationale: str, **args) -> dict:
        if self.trials >= self.budget:
            return {"error": "Trial budget exhausted. Analyse and finish."}
        c, r = self.exp.control, self.exp.record
        if c.name not in args:
            return {"error": f"run_trial needs {c.name} ({c.description})"}
        value = c.clamp(args[c.name])
        secs = r.clamp(args.get(r.name, r.hi))
        self.trials += 1
        n = self.trials
        self.nb.plan(n, self.exp.describe(value), secs, rationale)
        print(f"\n=== Trial {n}/{self.budget}: {self.exp.describe(value)} for {secs:.0f}s ===\n{rationale}")
        status.update(trial=n, budget=self.budget, setting=self.exp.describe(value), rationale=rationale)
        self.releaser.setup(value, seconds=secs + 1.0)
        raw = self.recorder.record(self.dir / f"raw{n:02d}.mp4", secs + 1.0, on_start=self.releaser.release)
        if self.mock is None:
            # SAM bills per frame: refuse a clip that would break the cap before paying for it.
            from .video import probe
            est = 2 * config.sam_cost(probe(raw).n_frames) + 0.10   # forward + backfill + technician
            if self.spent_usd + est > self.max_usd:
                msg = (f"Not measured: this clip would cost about ${est:.2f} and the session has "
                       f"${self.max_usd - self.spent_usd:.2f} left of its ${self.max_usd:.2f} cap.")
                print(f"\n[lab] {msg}")
                self.nb.write(f"**Trial {n} not measured (spending cap):** {msg}")
                return {"error": msg, "trials_left": self.budget - self.trials}
        status.write("measuring")
        res, _ = lab.measure_trial(raw, self.dir, trial=n, experiment=self.exp, control_value=value,
                                   mock=self.mock)
        self.nb.trial(res)
        status.write("thinking", message="Trial measured; the scientist is analysing it.")
        out = res.for_agent()
        out["trials_left"] = self.budget - self.trials
        out["usd_spent"] = round(self.spent_usd, 2)
        out["usd_cap"] = self.max_usd
        return out

    @property
    def spent_usd(self) -> float:
        """Everything live this session: scientist, technician and SAM frames."""
        return config.total_spend() - self._spend0

    def run_analysis(self, purpose: str, code: str) -> dict:
        output = run_python(code, self.dir)
        self.nb.analysis(purpose, code, output)
        return {"output": output}

    def log_finding(self, claim: str, evidence: str, confidence: str) -> dict:
        self.nb.finding(claim, evidence, confidence)
        return {"logged": True}

    # ---- loop --------------------------------------------------------------

    def run(self) -> str:
        status.set_session(self.dir)
        status.update(budget=self.budget, usd_cap=self.max_usd, usd_spent=0.0)
        status.write("thinking", message="The scientist is planning the first trial.")
        try:
            report = self._run()
        except BaseException as e:
            status.write("stopped", message=f"Session stopped: {type(e).__name__}: {e}")
            raise
        status.write("done", message=report or status.current().get("message")
                     or "Session ended without a final report.")
        return report

    def _run(self) -> str:
        client = config.client()
        for step in range(self.max_steps):
            if self.spent_usd >= self.max_usd:
                note = (f"Stopped at the spending cap: ${self.spent_usd:.2f} spent on the scientist, "
                        f"technician and SAM (cap ${self.max_usd:.2f}).")
                print(f"\n[scientist] {note}")
                self.nb.conclude(note)
                status.write("thinking", message=note)
                return ""
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=config.SCIENTIST_MODEL, messages=self.messages, tools=self.tools,
                max_tokens=16000, reasoning_effort=self.effort,
            )
            cost = config.chat_cost(config.SCIENTIST_MODEL, getattr(resp, "usage", None))
            config.record_spend("scientist", cost)
            print(f"[step {step + 1}] {time.perf_counter() - t0:.1f}s, ${cost:.4f} "
                  f"(session ${self.spent_usd:.2f}: " + ", ".join(
                      f"{k} ${v:.2f}" for k, v in config.SPEND.items()) + ")")
            msg = resp.choices[0].message
            self.messages.append(msg.model_dump(exclude_none=True))
            status.update(usd_spent=round(self.spent_usd, 2))
            if msg.content:
                print(f"\n[scientist] {msg.content}")
                status.write("thinking", message=msg.content)
            if not msg.tool_calls:
                self.messages.append({"role": "user", "content":
                                      "Continue: call a tool (run_trial, run_analysis, log_finding or finish)."})
                continue
            for call in msg.tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    result = {"error": f"bad JSON arguments: {e}"}
                else:
                    if name == "finish":
                        self.nb.conclude(args.get("report", ""))
                        print(f"\n[scientist] FINAL REPORT\n{args.get('report', '')}")
                        return args.get("report", "")
                    fn = getattr(self, name, None)
                    try:
                        result = fn(**args) if fn else {"error": f"unknown tool {name}"}
                    except Exception as e:  # noqa: BLE001 - report lab failures to the scientist
                        result = {"error": f"{type(e).__name__}: {e}"}
                print(f"[{name}] -> {json.dumps(result, default=str)[:600]}")
                self.messages.append({"role": "tool", "tool_call_id": call.id,
                                      "content": json.dumps(result, default=str)})
        self.nb.conclude("Stopped at the step limit before a final report.")
        return ""
