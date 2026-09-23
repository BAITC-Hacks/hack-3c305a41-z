"""Auditable priority contributions, normalized to 0–1."""
from typing import Any
import pandas as pd


def rank_nodes(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = frame.copy()
    settings = config["priority"]
    inputs = {"role": df.role.map(settings["role_values"]), "sources": df.norm_n_seed_sources,
              "volume": df.norm_in_kzt, "brokerage": df.norm_betweenness}
    columns = []
    for name, values in inputs.items():
        column = f"priority_{name}"
        df[column] = values * settings["weights"][name]
        columns.append(column)
    df["priority_score"] = df[columns].sum(axis=1)
    df["why"] = [f"Вклад: роль {r.priority_role:.3f}, seed {r.priority_sources:.3f}, вход {r.priority_volume:.3f}, посредничество {r.priority_brokerage:.3f}. Приоритет проверки, не оценка виновности."
                 for r in df.itertuples(index=False)]
    ordered = df.sort_values(["priority_score", "gid"], ascending=[False, True])
    rank = pd.Series(range(1, len(ordered) + 1), index=ordered.gid)
    df["rank"] = df.gid.map(rank)
    top = df.sort_values("rank").head(settings["top_count"])[["rank", "gid", "role", "priority_score", "why"]].reset_index(drop=True)
    return df, top
