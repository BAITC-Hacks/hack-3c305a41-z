"""Eligibility-gated, interpretable role scoring."""
from typing import Any
import numpy as np
import pandas as pd


def assign_roles(features: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    df = features.copy()
    cfg = config["roles"]
    observed_balance = ~df.is_seed & ~df.truncated_by_depth & (df.in_kzt > 0)
    gates = {}
    scores = {}
    c = cfg["consolidator"]
    gates["consolidator"] = observed_balance & (df.in_deg >= c["min_in_degree"]) & (df.pass_through <= c["max_pass_through"]) & (df.in_concentration <= c["max_concentration"])
    w = c["weights"]
    scores["consolidator"] = w["breadth"] * df.norm_in_deg + w["diversity"] * (1 - df.in_concentration) + w["retention"] * (1 - df.pass_through).clip(0, 1)
    c = cfg["distributor"]
    gates["distributor"] = df.out_deg >= c["min_out_degree"]
    w = c["weights"]
    scores["distributor"] = w["breadth"] * df.norm_out_deg + w["volume"] * df.norm_out_kzt
    c = cfg["transit"]
    gates["transit"] = observed_balance & (df.out_deg > 0) & df.pass_through.between(c["min_ratio"], c["max_ratio"])
    tolerance = np.where(df.pass_through < 1, 1 - c["min_ratio"], c["max_ratio"] - 1)
    balance = (1 - (df.pass_through - 1).abs() / tolerance).clip(0, 1)
    w = c["weights"]
    scores["transit"] = w["balance"] * balance + w["volume"] * df.norm_in_kzt
    c = cfg["terminal"]
    # Fan-in retention is a more specific pattern than merely having no outflow.
    gates["terminal"] = observed_balance & (df.out_deg == 0) & (df.in_kzt >= c["min_in_kzt"]) & ~gates["consolidator"]
    w = c["weights"]
    scores["terminal"] = w["no_outgoing"] + w["volume"] * df.norm_in_kzt
    c = cfg["coordinator"]
    gates["coordinator"] = ~df.is_seed & (df.out_deg > 0) & (df.in_deg > 0) & (df.depth >= c["min_depth"]) & (df.n_seed_sources >= c["min_seed_sources"]) & (df.norm_betweenness >= c["min_normalized_betweenness"])
    w = c["weights"]
    scores["coordinator"] = w["brokerage"] * df.norm_betweenness + w["sources"] * df.norm_n_seed_sources
    order = cfg["tie_order"]
    for role in order:
        df[f"eligible_{role}"] = gates[role]
        df[f"score_{role}"] = scores[role].where(gates[role], 0).fillna(0).clip(0, 1)
    matrix = df[[f"score_{r}" for r in order]].to_numpy()
    # Stable sorting provides an explicit tie-breaker independent of input order.
    sorted_indices = np.argsort(-matrix, axis=1, kind="stable")
    best = matrix[np.arange(len(df)), sorted_indices[:, 0]]
    second = matrix[np.arange(len(df)), sorted_indices[:, 1]]
    supported = best >= cfg["minimum_score"]
    df["role"] = np.where(supported, np.array(order)[sorted_indices[:, 0]], "peripheral")
    # Fallback score is capped and zero for isolates; it is never a claim of innocence.
    df["role_score"] = np.where(supported, best, np.where(df.is_isolated, 0, cfg["fallback_cap"] * (1 - best)))
    df["best_specialized_role"] = np.where(best > 0, np.array(order)[sorted_indices[:, 0]], "none")
    df["best_specialized_score"] = best
    df["role_runner_up"] = np.where(second > 0, np.array(order)[sorted_indices[:, 1]], "none")
    df["runner_up_score"] = second
    df["score_margin"] = best - second
    return df
