"""Observed-data limitations and concrete requests for further review."""
import pandas as pd


def add_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    limitations, requests = [], []
    for row in df.itertuples(index=False):
        flags, follow_up = [], []
        if row.truncated_by_depth:
            flags.append("truncated_at_hop_4")
            follow_up.append("Запросить исходящие за границей обхода")
        if row.is_seed:
            flags.append("seed_inflows_incomplete")
            follow_up.append("Запросить полные входящие seed")
        if row.is_isolated:
            flags.append("no_observed_edges")
            follow_up.append("Уточнить переводы вне периода, банка и порога выборки")
        if not flags:
            flags.append("partial_bank_month")
            follow_up.append("Сверить полную выписку и более широкий период")
        limitations.append(";".join(flags))
        requests.append("; ".join(follow_up))
    df["data_limitation"] = limitations
    df["next_request"] = requests
    return df
