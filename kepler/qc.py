"""Muse Spark as lab technician.

Before any number is trusted, Muse Spark watches the raw clip (native video
understanding) and answers in JSON: is this run usable, what went wrong if
not, and what short noun phrase should SAM 3.1 track. This is the pattern
Meta's docs recommend: Muse Spark decides *what* to segment, SAM 3.1 does
the pixels.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config

QC_PROMPT = """You are the lab technician in an automated physics lab. This clip is one
run of an experiment: {experiment}

Watch the whole clip and check it before it is measured. {rules}

Reply with JSON only, no prose, exactly these keys:
{{
  "usable": true or false,
  "issues": ["short description of each problem, with the time in seconds"],
  "release_time_s": number or null,   // when the object is let go
  "track_phrase": "a 1-3 word noun phrase naming the moving object, e.g. 'orange ball'",
  "reference_phrase": "noun phrase for an object of known size if one is visible, else null",
  "notes": "one sentence on anything a scientist should know"
}}"""

# The pendulum's rules (the original prompt); experiments pass their own.
DEFAULT_RULES = """Reject the run if any of
these happen: the moving object hits anything, a hand or person touches it after
release, the string goes slack or the object spins wildly, the object leaves the
frame, the camera moves, or the lighting changes sharply."""


def prompt(experiment: str, rules: str | None = None) -> str:
    return QC_PROMPT.format(experiment=experiment, rules=rules or DEFAULT_RULES)


@dataclass
class QCReport:
    usable: bool
    issues: list[str] = field(default_factory=list)
    release_time_s: float | None = None
    track_phrase: str | None = None
    reference_phrase: str | None = None
    notes: str = ""
    raw: str = ""
    checked: bool = True     # False: no parsable verdict after a retry ("unchecked")
    usage: dict = field(default_factory=dict)


def _json_block(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"No JSON in model reply: {text[:200]}")
    return json.loads(m.group(0))


def cache_path(clip: Path, experiment: str, model: str | None = None, rules: str | None = None) -> Path:
    from .sam import content_hash

    model = model or config.QC_MODEL
    key = f"{content_hash(clip)}:{model}:{experiment}" + (f":{rules}" if rules else "")
    h = hashlib.sha1(key.encode()).hexdigest()[:10]
    return clip.with_suffix(f".qc-{h}.txt")


def check_clip(file_id: str | None, experiment: str, *, cache: Path | None = None,
               model: str | None = None, attempts: int = 2, rules: str | None = None) -> QCReport:
    """Ask the technician for a verdict. An empty or unparsable reply is retried once;
    after that the clip is "unchecked", which callers must treat as not usable."""
    if cache is not None and cache.exists():
        return parse_report(cache.read_text(encoding="utf-8"))
    client = config.client()
    rep = None
    for _ in range(attempts):
        resp = client.responses.create(
            model=model or config.QC_MODEL,
            max_output_tokens=4000,
            reasoning={"effort": "low"},
            input=[{
                "type": "message", "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt(experiment, rules)},
                    {"type": "input_video", "file_id": file_id},
                ],
            }],
        )
        text = resp.output_text or ""
        rep = parse_report(text)
        u = getattr(resp, "usage", None)
        if u is not None:
            rep.usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens}
            config.record_spend("technician", config.response_cost(model or config.QC_MODEL, rep.usage))
        if rep.checked:
            if cache is not None:
                cache.write_text(text, encoding="utf-8")
            return rep
    return rep


def parse_report(text: str) -> QCReport:
    try:
        d = _json_block(text)
    except Exception as e:  # noqa: BLE001
        why = "empty QC reply" if not text.strip() else f"QC reply unparsable: {e}"
        return QCReport(usable=False, issues=[why], raw=text, checked=False)
    return QCReport(
        usable=bool(d.get("usable", True)),
        issues=list(d.get("issues") or []),
        release_time_s=d.get("release_time_s"),
        track_phrase=d.get("track_phrase"),
        reference_phrase=d.get("reference_phrase"),
        notes=d.get("notes") or "",
        raw=text,
    )
