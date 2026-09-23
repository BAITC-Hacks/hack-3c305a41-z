"""The must-have list from the brief must keep passing as the code changes."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.compliance import main

ROOT = Path(__file__).resolve().parents[1]


def test_published_results_satisfy_the_brief(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["compliance", "--out", str(ROOT / "out"), "--root", str(ROOT)])
    assert main() == 0
