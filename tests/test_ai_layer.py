"""The optional layer must never change what a plain run produces."""
from pathlib import Path

import pytest

from run import run
from src.ai import agents as roster
from src.ai.client import LLMUnavailable, build_client
from src.export import file_hash

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {"api_key_env": "OPENAI_API_KEY", "timeout_seconds": 30}


def test_missing_key_is_reported_not_raised_as_crash(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMUnavailable):
        build_client(SETTINGS)


def test_llm_flag_without_key_produces_identical_artifacts(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    plain = run(ROOT / "data", tmp_path / "plain", ROOT / "config.yaml")
    enriched = run(ROOT / "data", tmp_path / "llm", ROOT / "config.yaml", llm=True)
    assert plain["llm_used"] is False and enriched["llm_used"] is False
    for name in ("nodes_roles.csv", "clusters.csv", "top_nodes.csv"):
        assert file_hash(tmp_path / "plain" / name) == file_hash(tmp_path / "llm" / name)
    assert not (tmp_path / "llm" / "llm_review.json").exists()


def test_agent_links_form_a_review_stage():
    links = roster.graph_edges()
    assert {"from": "evidence", "to": "priority"} in links
    # Every writer must reach the critic: no text is published without a compliance pass.
    reviewed = {link["from"] for link in links if link["to"] == "critic"}
    assert reviewed == {writer.id for writer in roster.WRITERS}


def test_writer_prompts_forbid_guilt_claims():
    for agent in roster.WRITERS:
        assert "виновность" in agent.system and "гипотез" in agent.system
        assert agent.max_chars <= 300
    assert roster.EVIDENCE.max_chars == 200  # export contract from the brief
