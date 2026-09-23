"""Validate output invariants and publish a consistent bundle."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import numpy as np
import pandas as pd

NODE_COLUMNS = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]
CLUSTER_COLUMNS = ["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"]
TOP_COLUMNS = ["rank", "gid", "role", "priority_score", "why"]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_outputs(nodes: pd.DataFrame, clusters: pd.DataFrame, top: pd.DataFrame, input_nodes: pd.DataFrame, config: dict[str, Any]) -> None:
    for frame, required in ((nodes, NODE_COLUMNS), (clusters, CLUSTER_COLUMNS), (top, TOP_COLUMNS)):
        if not set(required).issubset(frame.columns) or frame[required].isna().any().any():
            raise ValueError("Незаполненная схема обязательной выгрузки")
    if nodes.gid.duplicated().any() or set(nodes.gid) != set(input_nodes.gid):
        raise ValueError("Выходные gid не совпадают с входом")
    if not set(nodes.role).issubset(set(config["roles"]["tie_order"]) | {"peripheral"}):
        raise ValueError("Неизвестная роль")
    if not np.isfinite(nodes[["role_score", "priority_score"]]).all().all() or not nodes[["role_score", "priority_score"]].apply(lambda s: s.between(0, 1)).all().all():
        raise ValueError("Скор вне 0–1")
    if not nodes.evidence.str.len().between(1, config["export"]["evidence_max_chars"]).all():
        raise ValueError("Некорректная длина evidence")
    if not clusters.set_index("cluster_id").n_nodes.sort_index().equals(nodes.groupby("cluster_id").size().sort_index().rename("n_nodes")):
        raise ValueError("Размеры кластеров не согласованы")
    if nodes.loc[nodes.truncated_by_depth, "role"].eq("terminal").any():
        raise ValueError("Обрезанный узел не может быть terminal")
    if len(top) != min(config["priority"]["top_count"], len(nodes)) or top.gid.duplicated().any():
        raise ValueError("Неверный размер top_nodes")
    if top["rank"].tolist() != list(range(1, len(top) + 1)) or not top.priority_score.is_monotonic_decreasing:
        raise ValueError("Неверный порядок top_nodes")
    if not np.allclose(nodes.priority_score, nodes[["priority_role", "priority_sources", "priority_volume", "priority_brokerage"]].sum(axis=1)):
        raise ValueError("Вклады не складываются в приоритет")


def write_outputs(nodes: pd.DataFrame, clusters: pd.DataFrame, top: pd.DataFrame, edges: pd.DataFrame, resilience: pd.DataFrame, out: Path, manifest: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".export-", dir=out.parent) as temporary:
        staging = Path(temporary)
        ordered = nodes.sort_values("gid").reset_index(drop=True)
        ordered[NODE_COLUMNS + [c for c in ordered if c not in NODE_COLUMNS]].to_csv(staging / "nodes_roles.csv", index=False, encoding="utf-8", lineterminator="\n", float_format="%.12g")
        clusters[CLUSTER_COLUMNS].to_csv(staging / "clusters.csv", index=False, encoding="utf-8", lineterminator="\n", float_format="%.12g")
        top[TOP_COLUMNS].to_csv(staging / "top_nodes.csv", index=False, encoding="utf-8", lineterminator="\n", float_format="%.12g")
        resilience.to_csv(staging / "resilience.csv", index=False, encoding="utf-8", lineterminator="\n", float_format="%.12g")
        ordered.to_parquet(staging / "node_features.parquet", index=False)
        edges.to_parquet(staging / "viewer_edges.parquet", index=False)
        manifest["artifacts"] = {p.name: file_hash(p) for p in sorted(staging.iterdir())}
        (staging / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        # Manifest is replaced last. The viewer checks hashes and refuses mixed runs.
        for path in sorted(staging.iterdir(), key=lambda p: p.name == "run_manifest.json"):
            os.replace(path, out / path.name)
