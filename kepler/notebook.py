"""The lab notebook: every decision, run, analysis and finding, in order.

Markdown, so it renders on GitHub and can be published as is. Every number
in it links back to the clip, observations and code that produced it.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path


class Notebook:
    def __init__(self, session_dir: str | Path, title: str, setting: str | None = None):
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "NOTEBOOK.md"
        if not self.path.exists():
            head = f"# {title}\n\n_Session started {dt.datetime.now():%Y-%m-%d %H:%M}_\n\n"
            if setting:
                head += f"> **Setting:** {setting}\n\n"
            self.path.write_text(head, encoding="utf-8")

    def write(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n\n")

    def plan(self, n: int, setting: str, seconds: float, rationale: str) -> None:
        self.write(f"## Trial {n}: {setting}, record {seconds:.0f} s\n\n"
                   f"**Why this experiment:** {rationale}")

    def trial(self, res) -> None:
        rel = lambda p: Path(p).name if p else ""
        status = {"pass": "passed", "rejected": "flagged", "overruled": "flagged, overruled by evidence",
                  "unchecked": "no verdict (unchecked)", "skipped": "not run"}.get(res.qc_status, res.qc_status)
        lines = [f"**Verdict:** {'usable' if res.usable else 'REJECTED'}. "
                 f"Technician: {status}"
                 + (f"; issues: {'; '.join(res.qc_issues)}" if res.qc_issues else "")
                 + (f". {res.qc_notes}" if res.qc_notes else ""),
                 f"**Tracked ({res.tracker}):** '{res.track_phrase}'" + (f", {res.summary}" if res.summary else "")]
        for family, fits in res.laws.items():
            lines.append(f"**{family.capitalize()} fits (this trial):**\n\n" + "\n".join(f"- {s}" for s in fits))
        if res.evidence_failures:
            lines.append("**Evidence checks failed:** " + "; ".join(res.evidence_failures))
        warnings = [w for w in res.warnings if not w.startswith("Evidence checks failed")]  # shown above
        if warnings:
            lines.append("**Warnings:** " + " ".join(warnings))
        if res.plot:
            lines.append(f"![trial {res.trial}]({rel(res.plot)})")
        if res.overlay:
            lines.append(f"Overlay video: `{rel(res.overlay)}`")
        self.write("\n\n".join(lines))

    def analysis(self, purpose: str, code: str, output: str) -> None:
        # Code folds away so the notebook reads as decisions and results; it is all still there.
        self.write(f"### Analysis: {purpose}\n\n<details><summary>Code ({len(code.strip().splitlines())} lines)"
                   f"</summary>\n\n```python\n{code.strip()}\n```\n\n</details>\n\n"
                   f"```text\n{output.strip()[:4000]}\n```")

    def finding(self, claim: str, evidence: str, confidence: str) -> None:
        self.write(f"> **Finding ({confidence}):** {claim}\n>\n> _Evidence:_ {evidence}")

    def conclude(self, text: str) -> None:
        self.write(f"## Conclusion\n\n{text}")
