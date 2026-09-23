"""What happens to the network when the top nodes are taken out.

This answers the question the ranking exists for. It also checks the ranking
against chance: removing N nodes we picked is compared with removing N random
ones, averaged over several seeded draws. If our order were uninformative the
two curves would sit on top of each other — the gap is the evidence that the
priority list points at structure rather than at big numbers.
"""
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


def _fragment(graph: nx.Graph, removed: list[int], edges: pd.DataFrame, seeds: set[int]) -> dict[str, Any]:
    remaining = graph.copy()
    remaining.remove_nodes_from(removed)
    components = sorted(nx.connected_components(remaining), key=len, reverse=True)
    largest = components[0] if components else set()
    survived = edges[edges.src.isin(largest) & edges.dst.isin(largest)].sum_kzt.sum()
    total = edges.sum_kzt.sum()
    return {"removed": len(removed),
            "largest_component": len(largest),
            "components_min2": sum(len(group) > 1 for group in components),
            "seeds_outside_largest": len(seeds - largest - set(removed)),
            "turnover_share_in_largest": float(survived / total) if total else 0.0}


def fragmentation(graph: nx.DiGraph, frame: pd.DataFrame, edges: pd.DataFrame,
                  config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    settings = config["resilience"]
    undirected = graph.to_undirected()
    undirected.add_nodes_from(frame.gid)  # isolated clients are part of the picture too
    seeds = set(frame.loc[frame.is_seed, "gid"])
    ranked = frame.sort_values(["priority_score", "gid"], ascending=[False, True]).gid.tolist()
    pool = frame.gid.tolist()

    rows = []
    for count in settings["steps"]:
        if count > len(pool):
            continue
        rows.append({"strategy": "priority", "trial": 0, **_fragment(undirected, ranked[:count], edges, seeds)})
        for trial in range(settings["random_trials"]):
            # Seeded per trial and per step: the comparison must reproduce exactly.
            generator = np.random.default_rng(config["random_seed"] + trial)
            draw = list(generator.permutation(pool)[:count])
            rows.append({"strategy": "random", "trial": trial, **_fragment(undirected, draw, edges, seeds)})

    table = pd.DataFrame(rows)
    averaged = (table.groupby(["strategy", "removed"], as_index=False)
                .agg({"largest_component": "mean", "components_min2": "mean",
                      "seeds_outside_largest": "mean", "turnover_share_in_largest": "mean"}))

    headline = settings["headline_step"]
    picked = averaged[(averaged.strategy == "priority") & (averaged.removed == headline)]
    chance = averaged[(averaged.strategy == "random") & (averaged.removed == headline)]
    baseline = averaged[averaged.removed == 0].largest_component.max()
    summary = {"headline_step": headline, "baseline_largest_component": int(baseline)}
    if not picked.empty and not chance.empty:
        ours, theirs = picked.iloc[0], chance.iloc[0]
        summary |= {"priority_largest_component": int(ours.largest_component),
                    "random_largest_component": round(float(theirs.largest_component), 1),
                    "priority_fragments": round(float(ours.components_min2), 1),
                    "random_fragments": round(float(theirs.components_min2), 1),
                    "priority_turnover_share": round(float(ours.turnover_share_in_largest), 4),
                    "random_turnover_share": round(float(theirs.turnover_share_in_largest), 4),
                    "advantage_nodes_detached": int(theirs.largest_component - ours.largest_component)}
    return averaged, summary
