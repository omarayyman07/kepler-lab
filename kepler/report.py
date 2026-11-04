"""A judge-readable HTML report of one session, built from its NOTEBOOK.md and plots.

    python -m kepler report sessions/<name>        ->  sessions/<name>/report.html

The notebook is the record; the report only re-presents it. Nothing is computed
or summarised here, so every number on the page is one the session wrote down.
Provenance comes from the notebook's Setting line: SIMULATED, REAL, or (for
sessions that predate it) "not recorded" — never assumed real.

The report is one self-contained file: plots are embedded, overlay videos are
linked relative to the session folder.
"""

from __future__ import annotations

import base64
import html
import re
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Notebook parsing (also used by scripts/make_demo.py)
# --------------------------------------------------------------------------


@dataclass
class FitLine:
    name: str
    delta_bic: str
    rms: str
    params: str
    raw: str


@dataclass
class Analysis:
    purpose: str
    code: str = ""
    output: str = ""


@dataclass
class Finding:
    confidence: str
    claim: str
    evidence: str = ""
    after_trial: int = 0


@dataclass
class Trial:
    n: int
    heading: str                      # e.g. "release at 75°, record 90 s"
    why: str = ""
    verdict: str = ""
    tracked: str = ""
    evidence_failed: str = ""
    warnings: str = ""
    fits: list[tuple[str, list[FitLine]]] = field(default_factory=list)
    plot: str = ""
    overlay: str = ""
    analyses: list[Analysis] = field(default_factory=list)


@dataclass
class Notebook:
    title: str = ""
    started: str = ""
    setting: str = ""
    trials: list[Trial] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    conclusion: str = ""
    preamble_analyses: list[Analysis] = field(default_factory=list)

    @property
    def provenance(self) -> str:
        s = self.setting.upper()
        if "SIMULATED" in s:
            return "simulated"
        if s.startswith("REAL"):
            return "real"
        return "unknown"

    @property
    def mock_tracking(self) -> bool:
        return "offline colour tracker" in self.setting or any("not SAM" in t.tracked for t in self.trials)


_FIT = re.compile(r"^(?P<name>.+?): ΔBIC=(?P<d>\S+)\s+rms=(?P<rms>\S+)\s*(?P<params>.*)$")


def _fit_line(s: str) -> FitLine:
    m = _FIT.match(s)
    if not m:
        return FitLine(s, "", "", "", s)
    return FitLine(m["name"], m["d"], m["rms"], m["params"], s)


def parse_notebook(text: str) -> Notebook:
    nb = Notebook()
    lines = text.splitlines()
    trial: Trial | None = None
    analysis: Analysis | None = None
    section = ""            # "", "trial", "conclusion"
    fence: str | None = None
    fence_buf: list[str] = []
    fits_family: str | None = None
    conclusion: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        # fenced code: belongs to the current analysis (python = code, text = output)
        if fence is not None:
            if ln.startswith("```"):
                body = "\n".join(fence_buf)
                if analysis is not None:
                    if fence == "python":
                        analysis.code = body
                    else:
                        analysis.output = body
                elif section == "conclusion":
                    conclusion.append(f"```\n{body}\n```")
                fence, fence_buf = None, []
            else:
                fence_buf.append(ln)
            i += 1
            continue
        if ln.startswith("```"):
            fence = ln[3:].strip() or "text"
            i += 1
            continue

        if ln.startswith("# ") and not nb.title:
            nb.title = ln[2:].strip()
        elif ln.startswith("_Session started"):
            nb.started = ln.strip("_ ").replace("Session started ", "")
        elif ln.startswith("> **Setting:**"):
            nb.setting = ln.split("**Setting:**", 1)[1].strip()
        elif m := re.match(r"^## Trial (\d+): (.*)$", ln):
            trial = Trial(int(m[1]), m[2].strip())
            nb.trials.append(trial)
            section, analysis, fits_family = "trial", None, None
        elif ln.startswith("## Conclusion"):
            section, trial, analysis = "conclusion", None, None
        elif m := re.match(r"^### Analysis: (.*)$", ln):
            analysis = Analysis(m[1].strip())
            (trial.analyses if trial else nb.preamble_analyses).append(analysis)
            fits_family = None
        elif ln.startswith("> **Finding"):
            m = re.match(r"^> \*\*Finding \(([^)]*)\):\*\* (.*)$", ln)
            f = Finding(m[1] if m else "", m[2] if m else ln[2:], after_trial=trial.n if trial else 0)
            j = i + 1
            while j < len(lines) and lines[j].startswith(">"):
                body = lines[j][1:].strip()
                if body.startswith("_Evidence:_"):
                    f.evidence = body.replace("_Evidence:_", "", 1).strip()
                elif body and f.evidence:
                    f.evidence += " " + body
                j += 1
            nb.findings.append(f)
            i = j
            continue
        elif section == "conclusion":
            conclusion.append(ln)
        elif section == "trial" and trial is not None and analysis is None:
            if m := re.match(r"^\*\*(.+?):\*\*\s*(.*)$", ln):
                key, val = m[1], m[2]
                fits_family = None
                if key == "Why this experiment":
                    trial.why = val
                elif key == "Verdict" or key.startswith("Technician check"):
                    trial.verdict = val
                elif key.startswith("Tracked"):
                    trial.tracked = key[len("Tracked"):].strip(" ()") + (": " if val else "") + val
                elif key == "Evidence checks failed":
                    trial.evidence_failed = val
                elif key == "Warnings":
                    trial.warnings = val
                elif key.endswith("fits (this trial)"):
                    fits_family = key.replace(" fits (this trial)", "")
                    trial.fits.append((fits_family, []))
            elif ln.startswith("- ") and fits_family:
                trial.fits[-1][1].append(_fit_line(ln[2:].strip()))
            elif m := re.match(r"^!\[[^\]]*\]\(([^)]+)\)", ln):
                trial.plot = m[1]
            elif m := re.match(r"^Overlay video: `([^`]+)`", ln):
                trial.overlay = m[1]
        i += 1
    nb.conclusion = "\n".join(conclusion).strip()
    return nb


def load(session: str | Path) -> Notebook:
    return parse_notebook((Path(session) / "NOTEBOOK.md").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

def _inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])_([^_]+)_(?![\w*])", r"<em>\1</em>", s)
    return s


def _paragraphs(text: str) -> str:
    """Model-written prose: paragraphs, numbered/bulleted lines, kept verbatim."""
    out, para = [], []

    def flush():
        if para:
            out.append("<p>" + "<br>".join(_inline(p) for p in para) + "</p>")
            para.clear()
    for ln in text.splitlines():
        if not ln.strip():
            flush()
        elif re.match(r"^[A-Z0-9 .,()'’\-–—]{6,}$", ln.strip()) and ln.strip() == ln.strip().upper():
            flush()
            out.append(f"<h4>{_inline(ln.strip())}</h4>")
        else:
            para.append(ln.strip())
    flush()
    return "\n".join(out)


def _img(session: Path, rel: str) -> str:
    p = session / rel
    if not rel or not p.exists():
        return ""
    data = base64.b64encode(p.read_bytes()).decode()
    return f'<img class="plot" alt="{html.escape(rel)}" src="data:image/png;base64,{data}">'


def _fits_table(family: str, fits: list[FitLine]) -> str:
    rows = []
    for k, f in enumerate(fits):
        cls = ' class="best"' if k == 0 else ""
        rows.append(f"<tr{cls}><td>{_inline(f.name)}</td><td class=num>{html.escape(f.delta_bic)}</td>"
                    f"<td class=num>{html.escape(f.rms)}</td><td class=params>{_inline(f.params)}</td></tr>")
    return (f'<div class="fits"><h4>{_inline(family.capitalize())}: competing laws, ranked by BIC</h4>'
            '<div class="scroll"><table><thead><tr><th>Law</th><th>ΔBIC</th><th>rms</th><th>Parameters</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div></div>')


PROVENANCE = {
    "simulated": ("SIMULATED DATA", "Exact-physics simulation, not real measurements. It rehearses the "
                  "agent and the analysis; the true values are in the setting below."),
    "real": ("REAL DATA", "Measured on a real apparatus."),
    "unknown": ("PROVENANCE NOT RECORDED", "This session predates provenance labels. Treat its numbers as "
                "unverified until the session folder is checked."),
}

CSS = """
:root{--bg:#f7f7f5;--fg:#1d1d1f;--muted:#5d5d63;--card:#fff;--line:#e3e3e0;--accent:#b8590b;
--good:#1f7a4d;--warn:#a15c00;--sim:#8a4b00;--simbg:#fff3e0;--real:#0f6b3f;--realbg:#e7f6ee;
--unk:#555;--unkbg:#eee;--code:#f1f1ee}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#141414;--fg:#ececec;--muted:#a3a3a8;
--card:#1d1d1f;--line:#2e2e31;--accent:#f0a050;--good:#5fd19a;--warn:#f0b050;--sim:#ffc27a;--simbg:#3a2a12;
--real:#7be0aa;--realbg:#12301f;--unk:#ccc;--unkbg:#2a2a2a;--code:#232326}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.9rem;margin:.2em 0}h2{font-size:1.35rem;margin:2em 0 .6em}h3{font-size:1.1rem;margin:0 0 .4em}
h4{font-size:.95rem;margin:1.2em 0 .4em;color:var(--muted);letter-spacing:.02em}
.sub{color:var(--muted)}.banner{border-radius:10px;padding:14px 16px;margin:18px 0;font-weight:600}
.banner small{display:block;font-weight:400;margin-top:4px}
.banner.simulated{background:var(--simbg);color:var(--sim)}.banner.real{background:var(--realbg);color:var(--real)}
.banner.unknown{background:var(--unkbg);color:var(--unk)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:12px 0}
.findings{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}
.finding .conf{font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;color:var(--accent);font-weight:700}
.finding p{margin:.4em 0}.finding details{font-size:.9rem;color:var(--muted)}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 10px}
.chip{font-size:.78rem;padding:2px 8px;border-radius:99px;border:1px solid var(--line);color:var(--muted)}
.chip.ok{color:var(--good);border-color:currentColor}.chip.bad{color:#c0392b;border-color:currentColor}
.chip.sim{color:var(--sim);border-color:currentColor}
.why{border-left:3px solid var(--accent);padding:2px 0 2px 12px;margin:10px 0}
.plot{width:100%;height:auto;border-radius:8px;border:1px solid var(--line);background:#fff;margin-top:8px}
video{width:100%;border-radius:8px;margin-top:8px;background:#000}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}
td.num{font-variant-numeric:tabular-nums;white-space:nowrap}td.params{font-family:ui-monospace,Consolas,monospace;font-size:.8rem}
tr.best td{font-weight:600}code,pre{font-family:ui-monospace,Consolas,monospace;font-size:.82rem}
code{background:var(--code);padding:1px 4px;border-radius:4px}
pre{background:var(--code);padding:10px;border-radius:8px;overflow-x:auto;white-space:pre}
details>summary{cursor:pointer;color:var(--muted)}.analysis{margin:10px 0}
.warn{color:var(--warn)}footer{margin-top:48px;color:var(--muted);font-size:.85rem}
"""


def render_html(session: str | Path, nb: Notebook | None = None) -> str:
    session = Path(session)
    nb = nb or load(session)
    label, note = PROVENANCE[nb.provenance]
    e = html.escape
    parts = [f"<!doctype html><html lang=en><head><meta charset=utf-8>"
             f"<meta name=viewport content='width=device-width,initial-scale=1'>"
             f"<title>Kepler session report</title><style>{CSS}</style></head><body><main>",
             f"<p class=sub>Kepler · autonomous desk lab · session <code>{e(session.name)}</code>"
             f"{' · started ' + e(nb.started) if nb.started else ''}</p>",
             f"<h1>{e(nb.title.replace('Kepler lab notebook: ', '').capitalize() or 'Session')}</h1>",
             f'<div class="banner {nb.provenance}">{label}<small>{e(note)}</small></div>']
    if nb.setting:
        parts.append(f"<p class=sub><strong>Setting.</strong> {_inline(nb.setting)}</p>")

    if nb.findings:
        parts.append("<h2>What the scientist concluded</h2><div class=findings>")
        for f in nb.findings:
            parts.append(f'<div class="card finding"><div class=conf>{e(f.confidence)}</div>'
                         f"<p>{_inline(f.claim)}</p>"
                         + (f"<details><summary>Evidence</summary><p>{_inline(f.evidence)}</p></details>"
                            if f.evidence else "") + "</div>")
        parts.append("</div>")

    parts.append(f"<h2>How it got there: {len(nb.trials)} trial{'s' if len(nb.trials) != 1 else ''}</h2>")
    for a in nb.preamble_analyses:
        parts.append(_analysis_html(a))
    for t in nb.trials:
        usable = t.verdict.lower().startswith("usable")
        chips = [f'<span class="chip {"ok" if usable else "bad"}">{"usable" if usable else "rejected"}</span>']
        if m := re.search(r"Technician: ([^;.]+)", t.verdict):
            chips.append(f'<span class=chip>technician: {e(m[1].strip())}</span>')
        if "not SAM" in t.tracked:
            chips.append('<span class="chip sim">offline tracker, not SAM</span>')
        elif t.tracked:
            chips.append(f'<span class=chip>tracked by {e(t.tracked.split(":")[0])}</span>')
        parts.append(f'<section class=card><h3>Trial {t.n}: {_inline(t.heading)}</h3>'
                     f'<div class=chips>{"".join(chips)}</div>')
        if t.why:
            parts.append(f'<div class=why><strong>Why this trial.</strong> {_inline(t.why)}</div>')
        if t.tracked:
            parts.append(f"<p class=sub>{_inline(t.tracked)}</p>")
        if t.evidence_failed:
            parts.append(f"<p class=warn><strong>Evidence checks failed:</strong> {_inline(t.evidence_failed)}</p>")
        if t.warnings:
            parts.append(f"<p class=warn><strong>Warnings:</strong> {_inline(t.warnings)}</p>")
        parts.append(_img(session, t.plot))
        if t.overlay and (session / t.overlay).exists():
            parts.append(f'<video controls preload=none src="{e(t.overlay)}"></video>')
        for family, fits in t.fits:
            parts.append(_fits_table(family, fits))
        if t.analyses:
            parts.append(f"<h4>Analyses after this trial ({len(t.analyses)})</h4>")
            parts.extend(_analysis_html(a) for a in t.analyses)
        parts.append("</section>")

    if nb.conclusion:
        parts.append(f"<h2>Final report, verbatim</h2><div class=card>{_paragraphs(nb.conclusion)}</div>")
    parts.append(f"<footer>Generated from <code>NOTEBOOK.md</code> in <code>{e(session.name)}</code>. "
                 f"The report adds no numbers of its own. {e(label.capitalize())}.</footer></main></body></html>")
    return "\n".join(p for p in parts if p)


def _analysis_html(a: Analysis) -> str:
    e = html.escape
    out = f"<details class=analysis><summary>{_inline(a.purpose)}</summary>"
    if a.code:
        out += f"<pre><code>{e(a.code)}</code></pre>"
    if a.output:
        out += f"<p class=sub>Output</p><pre>{e(a.output)}</pre>"
    return out + "</details>"


def write_report(session: str | Path, out: str | Path | None = None) -> Path:
    session = Path(session)
    out = Path(out) if out else session / "report.html"
    out.write_text(render_html(session), encoding="utf-8")
    return out
