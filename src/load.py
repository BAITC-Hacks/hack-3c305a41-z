"""Validate raw organizer data before computing any conclusions."""
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMAS = {
    "nodes": ["gid", "depth", "is_seed"],
    "edges": ["src", "dst", "sum_kzt", "n_tx", "depth"],
    "transactions": ["src", "dst", "date", "sum_kzt"],
}


def load_data(folder: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frames = {}
    for name, columns in SCHEMAS.items():
        path = folder / f"{name}.parquet"
        if not path.is_file():
            raise ValueError(f"Нет входного файла: {path}")
        frame = pd.read_parquet(path)
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{name}: отсутствуют колонки {sorted(missing)}")
        if frame[columns].isna().any().any():
            raise ValueError(f"{name}: пропуски в обязательных полях")
        frames[name] = frame[columns].copy()
    nodes, edges, tx = frames["nodes"], frames["edges"], frames["transactions"]
    if nodes.empty:
        raise ValueError("nodes: пустая таблица узлов")
    for name, frame in frames.items():
        for column in set(frame.columns) & {"gid", "src", "dst", "depth", "n_tx"}:
            if not pd.api.types.is_integer_dtype(frame[column]):
                raise ValueError(f"{name}.{column}: ожидается целочисленный тип")
            if column in {"gid", "src", "dst"}:
                limits = np.iinfo(np.int64)
                if not frame[column].between(limits.min, limits.max).all():
                    raise ValueError(f"{name}.{column}: идентификатор вне int64")
                frame[column] = frame[column].astype("int64")
        if "sum_kzt" in frame:
            if not pd.api.types.is_numeric_dtype(frame.sum_kzt) or not np.isfinite(frame.sum_kzt).all() or (frame.sum_kzt <= 0).any():
                raise ValueError(f"{name}: суммы должны быть положительными конечными числами")
    if not pd.api.types.is_bool_dtype(nodes.is_seed):
        raise ValueError("nodes.is_seed: ожидается bool")
    if nodes.gid.duplicated().any() or edges.duplicated(["src", "dst"]).any():
        raise ValueError("Дубликат gid или агрегированной пары src/dst")
    for name, frame in (("edges", edges), ("transactions", tx)):
        if not set(frame.src).union(frame.dst).issubset(set(nodes.gid)):
            raise ValueError(f"{name}: конец ребра отсутствует в nodes")
    limit = config["data"]["max_depth"]
    if not nodes.depth.between(0, limit).all() or not edges.depth.between(1, limit).all():
        raise ValueError("Глубина вне допустимого диапазона обхода")
    if not ((nodes.depth == 0) == nodes.is_seed).all():
        raise ValueError("is_seed не согласуется с depth=0")
    if (edges.n_tx <= 0).any():
        raise ValueError("edges.n_tx должен быть положительным")
    tx["date"] = pd.to_datetime(tx.date, errors="raise")
    if tx.date.isna().any():
        raise ValueError("transactions.date содержит пустые даты")
    lower, upper = pd.Timestamp(config["data"]["period_start"]), pd.Timestamp(config["data"]["period_end"])
    if not tx.date.dt.normalize().between(lower, upper).all():
        raise ValueError("Транзакции вне настроенного периода")
    if (tx.sum_kzt < config["data"]["min_transaction_kzt"]).any():
        raise ValueError("Транзакции ниже объявленного порога включения")
    # Pair equality alone is insufficient: compare both amounts and counts.
    aggregated = tx.groupby(["src", "dst"], as_index=False).agg(tx_sum=("sum_kzt", "sum"), tx_count=("sum_kzt", "size"))
    joined = edges.merge(aggregated, on=["src", "dst"], how="outer", indicator=True, validate="one_to_one")
    if not (joined._merge == "both").all():
        raise ValueError("Пары edges и transactions не совпадают")
    if not np.isclose(joined.sum_kzt, joined.tx_sum, rtol=0, atol=config["data"]["amount_tolerance"]).all():
        raise ValueError("Суммы edges и transactions не совпадают")
    if not (joined.n_tx == joined.tx_count).all():
        raise ValueError("Количество переводов edges и transactions не совпадает")
    counts = {"nodes": len(nodes), "edges": len(edges), "transactions": len(tx), "seeds": int(nodes.is_seed.sum())}
    warnings = [f"{key}: фактически {counts[key]}, ориентир ТЗ {value}" for key, value in config["data"]["expected"].items() if counts[key] != value]
    report = {**counts, "sum_kzt": float(edges.sum_kzt.sum()), "warnings": warnings,
              "period_start": None if tx.empty else str(tx.date.min()), "period_end": None if tx.empty else str(tx.date.max()),
              "aggregates_match": True}
    return (nodes.sort_values("gid").reset_index(drop=True), edges.sort_values(["src", "dst"]).reset_index(drop=True),
            tx.sort_values(["date", "src", "dst", "sum_kzt"]).reset_index(drop=True), report)
