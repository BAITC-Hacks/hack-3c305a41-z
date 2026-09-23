"""Stable community IDs and summaries; turnover stays directed."""
from typing import Any
import networkx as nx
import pandas as pd
from src.graph import community_projection


def assign_clusters(frame: pd.DataFrame, graph: nx.DiGraph, config: dict[str, Any]) -> pd.DataFrame:
    projected = community_projection(graph)
    isolates = sorted(nx.isolates(projected))
    active = projected.subgraph(sorted(set(projected) - set(isolates))).copy()
    settings = config["clusters"]
    groups = nx.community.louvain_communities(active, weight="weight", resolution=settings["resolution"],
                                             threshold=settings["threshold"], seed=config["random_seed"]) if active.number_of_edges() else []
    groups += [{gid} for gid in isolates]
    groups = sorted(groups, key=lambda group: (-len(group), min(group)))
    mapping = {gid: cluster_id for cluster_id, group in enumerate(groups) for gid in group}
    df = frame.copy()
    df["cluster_id"] = df.gid.map(mapping).astype(int)
    return df


def summarize_clusters(frame: pd.DataFrame, edges: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    mapping = frame.set_index("gid").cluster_id
    assigned = edges.assign(source_cluster=edges.src.map(mapping), target_cluster=edges.dst.map(mapping))
    internal = assigned[assigned.source_cluster == assigned.target_cluster].groupby("source_cluster").sum_kzt.sum()
    rows = []
    for cluster_id, group in frame.groupby("cluster_id", sort=True):
        top = group.sort_values(["priority_score", "gid"], ascending=[False, True]).head(config["clusters"]["top_gids_count"])
        consolidators = int((group.role == "consolidator").sum())
        distributors = int((group.role == "distributor").sum())
        hypothesis = (f"Группа из {len(group)} узлов, seed {int(group.is_seed.sum())}; "
                      f"кандидаты: консолидация {consolidators}, распределение {distributors}. Связность требует проверки.")
        rows.append({"cluster_id": int(cluster_id), "n_nodes": len(group), "n_seed": int(group.is_seed.sum()),
                     "sum_kzt_internal": float(internal.get(cluster_id, 0)), "top_gids": ";".join(map(str, top.gid)), "hypothesis": hypothesis})
    return pd.DataFrame(rows)
