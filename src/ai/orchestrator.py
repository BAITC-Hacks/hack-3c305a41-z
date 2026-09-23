"""Runs the agent layer over finished numbers.

Order comes from the `consumes` links in `agents.py`: writers run in parallel,
then the critic reviews every draft and may send one revision back. Any failure
at any point keeps the deterministic text, so `--llm` can never produce a worse
artifact than a plain run.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from src.ai import agents as roster
from src.ai.client import LLMUnavailable, ask_json, build_client

NUMERIC_CONTEXT = ["role", "role_score", "in_deg", "out_deg", "in_kzt", "out_kzt", "pass_through",
                   "in_concentration", "n_seed_sources", "betweenness", "depth", "is_seed",
                   "truncated_by_depth", "data_limitation", "next_request"]
PRIORITY_CONTEXT = ["role", "priority_score", "priority_role", "priority_sources",
                    "priority_volume", "priority_brokerage", "rank"]


def _payload(row: pd.Series, columns: list[str], extra: dict[str, Any] | None = None) -> str:
    data = {c: row[c] for c in columns if c in row.index}
    data = {k: (v.item() if hasattr(v, "item") else v) for k, v in data.items()}
    return json.dumps({**data, **(extra or {})}, ensure_ascii=False, default=str)


def _review(client, settings, agent: roster.Agent, draft: str, source: str) -> dict[str, Any]:
    """Critic pass. A rejected draft is rewritten once; a failed review keeps the draft."""
    try:
        verdict = ask_json(client, settings, settings["models"]["critic"], roster.CRITIC.system,
                           f"Лимит символов: {agent.max_chars}\nИсходные данные: {source}\nТекст: {draft}")
    except LLMUnavailable:
        return {"verdict": "unchecked", "issues": [], "text": draft}
    revised = str(verdict.get("revised") or draft).strip()
    ok = verdict.get("verdict") == "ok"
    return {"verdict": "ok" if ok else "revise", "issues": list(verdict.get("issues") or []),
            "text": draft if ok else revised}


def _write_one(client, settings, agent: roster.Agent, key: Any, source: str, fallback: str) -> dict[str, Any]:
    record = {"agent": agent.name, "key": key, "offline": fallback, "used": "offline"}
    try:
        draft = str(ask_json(client, settings, settings["models"]["writer"], agent.system, source)[agent.output_key]).strip()
    except (LLMUnavailable, KeyError, TypeError) as error:
        record["error"] = str(error)
        record["final"] = fallback
        return record
    record["draft"] = draft
    review = _review(client, settings, agent, draft, source)
    record["critic"] = review["verdict"]
    record["issues"] = review["issues"]
    text = review["text"]
    # The export contract wins over the model: an over-long text is dropped, not truncated.
    if text and len(text) <= agent.max_chars:
        record["final"], record["used"] = text, "llm"
    else:
        record["final"] = fallback
        record["error"] = f"длина {len(text)} превышает лимит {agent.max_chars}"
    return record


def _run_agent(client, settings, agent, items) -> list[dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=settings["max_parallel"]) as pool:
        return list(pool.map(lambda item: _write_one(client, settings, agent, *item), items))


def enrich(nodes: pd.DataFrame, clusters: pd.DataFrame, top: pd.DataFrame,
           config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    settings = config["llm"]
    client = build_client(settings)
    nodes, clusters, top = nodes.copy(), clusters.copy(), top.copy()
    before = nodes[["gid", "role", "role_score", "priority_score"]].copy()

    selected = nodes.nsmallest(settings["top_nodes"], "rank")
    evidence_log = _run_agent(client, settings, roster.EVIDENCE,
                              [(r.gid, _payload(r, NUMERIC_CONTEXT), r.evidence) for _, r in selected.iterrows()])
    texts = {record["key"]: record["final"] for record in evidence_log}
    nodes["evidence"] = nodes.gid.map(texts).fillna(nodes.evidence)

    big = clusters[clusters.n_nodes >= settings["min_cluster_size"]]
    cluster_log = _run_agent(client, settings, roster.CLUSTER,
                             [(r.cluster_id, _payload(r, list(clusters.columns),
                                                      {"состав_ролей": nodes.loc[nodes.cluster_id == r.cluster_id, "role"].value_counts().to_dict()}),
                               r.hypothesis) for _, r in big.iterrows()])
    hypotheses = {record["key"]: record["final"] for record in cluster_log}
    clusters["hypothesis"] = clusters.cluster_id.map(hypotheses).fillna(clusters.hypothesis)

    merged = top.merge(nodes[["gid", *[c for c in PRIORITY_CONTEXT if c not in top.columns]]], on="gid", how="left")
    priority_log = _run_agent(client, settings, roster.PRIORITY,
                              [(r.gid, _payload(r, PRIORITY_CONTEXT, {"обоснование_роли": texts.get(r.gid, "")}), r.why)
                               for _, r in merged.iterrows()])
    reasons = {record["key"]: record["final"] for record in priority_log}
    top["why"] = top.gid.map(reasons).fillna(top.why)
    nodes["why"] = nodes.gid.map(reasons).fillna(nodes.why)

    log = evidence_log + cluster_log + priority_log
    # The layer is narrative only: any drift in the decisions would be a defect.
    if not before.equals(nodes[["gid", "role", "role_score", "priority_score"]]):
        raise ValueError("ИИ-слой изменил расчётные значения — недопустимо")
    summary = {"model_writer": settings["models"]["writer"], "model_critic": settings["models"]["critic"],
               "calls": len(log), "texts_from_model": sum(r["used"] == "llm" for r in log),
               "texts_kept_offline": sum(r["used"] == "offline" for r in log),
               "critic_revisions": sum(r.get("critic") == "revise" for r in log),
               "agent_links": roster.graph_edges(), "log": log}
    return nodes, clusters, top, summary
