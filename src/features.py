"""Observable node metrics; missing ratios remain missing."""
from typing import Any
import networkx as nx
import numpy as np
import pandas as pd


def normalize_positive(values: pd.Series, quantile: float) -> tuple[pd.Series, float]:
    positive = values[values > 0]
    scale = float(positive.quantile(quantile)) if len(positive) else 0.0
    normalized = (values / scale).clip(0, 1) if scale else pd.Series(0.0, index=values.index)
    return normalized, scale


def compute_features(graph: nx.DiGraph, nodes: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = nodes.copy()
    for direction in ("in", "out"):
        degree = graph.in_degree if direction == "in" else graph.out_degree
        for suffix, weight in (("deg", None), ("kzt", "sum_kzt"), ("tx", "n_tx")):
            frame[f"{direction}_{suffix}"] = frame.gid.map(dict(degree(weight=weight)))
        frame[f"avg_{direction}_tx"] = frame[f"{direction}_kzt"] / frame[f"{direction}_tx"].replace(0, np.nan)
    settings = config["features"]
    pagerank = nx.pagerank(graph, alpha=settings["pagerank_alpha"], tol=settings["pagerank_tolerance"], max_iter=settings["pagerank_max_iterations"], weight="sum_kzt")
    frame["pagerank"] = frame.gid.map(pagerank)
    sample = settings["betweenness_sample"]
    if sample is not None and (not isinstance(sample, int) or sample <= 0):
        raise ValueError("betweenness_sample должен быть положительным целым или null")
    k = min(sample, len(graph)) if sample is not None else None
    frame["betweenness"] = frame.gid.map(nx.betweenness_centrality(graph, k=k, weight=None, seed=config["random_seed"]))
    sources = dict.fromkeys(graph, 0)
    for seed in sorted(nodes.loc[nodes.is_seed, "gid"]):
        for gid in nx.descendants(graph, seed):
            sources[gid] += 1
    frame["n_seed_sources"] = frame.gid.map(sources)
    max_incoming = {gid: max((d["sum_kzt"] for _, _, d in graph.in_edges(gid, data=True)), default=0) for gid in graph}
    frame["in_concentration"] = frame.gid.map(max_incoming) / frame.in_kzt.replace(0, np.nan)
    frame["pass_through"] = frame.out_kzt / frame.in_kzt.replace(0, np.nan)
    frame["truncated_by_depth"] = (frame.depth == config["data"]["max_depth"]) & (frame.out_deg == 0)
    frame["is_isolated"] = (frame.in_deg + frame.out_deg) == 0
    components = sorted(nx.weakly_connected_components(graph), key=lambda group: (-len(group), min(group)))
    component_ids = {gid: i for i, group in enumerate(components) for gid in group}
    frame["component_id"] = frame.gid.map(component_ids)
    scales = {}
    for name in ("in_deg", "out_deg", "in_kzt", "out_kzt", "n_seed_sources", "betweenness"):
        frame[f"norm_{name}"], scales[name] = normalize_positive(frame[name], settings["normalization_quantile"])
    diagnostics = {"normalization_scales": scales, "betweenness_sample": k,
                   "weak_components_with_isolates": len(components),
                   "weak_components_without_isolates": sum(len(c) > 1 or graph.degree(next(iter(c))) > 0 for c in components),
                   "isolated_nodes": int(frame.is_isolated.sum()), "largest_component": len(components[0]),
                   "outside_largest_component": len(graph) - len(components[0]), "truncated_nodes": int(frame.truncated_by_depth.sum())}
    return frame, diagnostics
