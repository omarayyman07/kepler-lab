"""The published real results reproduce from their cached SAM output: no video, no API key."""

from pathlib import Path

import pytest

from kepler.replay import compare, replay

RESULTS = Path(__file__).resolve().parents[1] / "results"


@pytest.mark.parametrize("name", sorted(p.name for p in RESULTS.iterdir() if (p / "replay.json").exists()))
def test_published_result_reproduces(name, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)     # replay must never need the API
    data, m, _ = replay(RESULTS / name)
    assert compare(data["result"]["headline"], m.headline) == []
    assert m.summary == data["result"]["summary"]
