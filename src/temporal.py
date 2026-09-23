"""Timing signals from individual transactions.

The aggregated edges hide *when* money moved. Two nodes with the same monthly
turnover look identical until you ask whether the money sat there or left the
same week. These features are reported next to the role, not folded into it:
the role criteria stay the stable, auditable part of the solution.
"""
from typing import Any

import numpy as np
import pandas as pd


def compute_temporal(tx: pd.DataFrame, nodes: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    settings = config["temporal"]
    frame = tx.sort_values(["date", "src", "dst"], kind="stable")
    result = pd.DataFrame({"gid": nodes.gid.to_numpy()})

    # Activity profile over both directions, in one pass over the transactions.
    both = pd.concat([frame[["dst", "date", "sum_kzt"]].rename(columns={"dst": "gid"}),
                      frame[["src", "date", "sum_kzt"]].rename(columns={"src": "gid"})])
    daily = both.groupby(["gid", "date"], sort=False).sum_kzt.sum()
    per_day = daily.groupby("gid").agg(["size", "max", "mean"])
    result["active_days"] = result.gid.map(per_day["size"]).fillna(0).astype(int)
    # Burst: the busiest day measured against an even spread over the active days.
    result["burst_ratio"] = result.gid.map(per_day["max"] / per_day["mean"].replace(0, np.nan))

    # Synchronised inflows: several payers crediting the same node on one day.
    sync = frame.groupby(["dst", "date"], sort=False).src.nunique().groupby("dst").max()
    result["same_day_inflows"] = result.gid.map(sync).fillna(0).astype(int)

    arrivals = {gid: group.date.to_numpy() for gid, group in frame.groupby("dst", sort=False)}
    holds, shares = {}, {}
    for gid, group in frame.groupby("src", sort=False):
        received = arrivals.get(gid)
        if received is None:
            continue
        departures = group.date.to_numpy()
        # For each outgoing transfer: how long after the previous inflow it left.
        position = np.searchsorted(received, departures, side="right") - 1
        observed = position >= 0
        if not observed.any():
            continue
        gaps = (departures[observed] - received[position[observed]]) / np.timedelta64(1, "D")
        holds[gid] = float(np.median(gaps))
        total = float(group.sum_kzt.sum())
        if total:
            amounts = group.sum_kzt.to_numpy()[observed]
            shares[gid] = float(amounts[gaps <= settings["fast_pass_days"]].sum() / total)

    result["hold_days"] = result.gid.map(holds)
    result["fast_pass_share"] = result.gid.map(shares)
    result["fast_pass"] = result.fast_pass_share.fillna(0) >= settings["fast_pass_min_share"]
    result["synchronised_inflow"] = result.same_day_inflows >= settings["sync_min_payers"]
    return result


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    fast = frame[frame.fast_pass]
    return {"nodes_with_timing": int(frame.hold_days.notna().sum()),
            "fast_pass_nodes": int(frame.fast_pass.sum()),
            "fast_pass_transit_nodes": int((fast.role == "transit").sum()) if "role" in fast else 0,
            "synchronised_inflow_nodes": int(frame.synchronised_inflow.sum()),
            "median_hold_days": float(frame.hold_days.median(skipna=True)) if frame.hold_days.notna().any() else None}
