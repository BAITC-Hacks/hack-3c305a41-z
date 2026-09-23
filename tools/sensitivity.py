"""Threshold sensitivity: how many nodes change role when a threshold moves.

Run it to answer the jury question "почему порог именно такой". Offline, no API.

    python tools/sensitivity.py --data data
"""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import load_config
from src.features import compute_features
from src.graph import build_graph
from src.load import load_data
from src.roles import assign_roles

# Each sweep keeps the pipeline untouched and only moves one number at a time.
SWEEPS = {
    "roles.consolidator.min_in_degree": [2, 3, 4, 5, 8],
    "roles.consolidator.max_pass_through": [0.4, 0.6, 0.8],
    "roles.distributor.min_out_degree": [5, 10, 20, 60],
    "roles.terminal.min_in_kzt": [5_000, 50_000, 250_000, 1_000_000],
    "roles.coordinator.min_normalized_betweenness": [0.5, 0.7, 0.9],
    "roles.minimum_score": [0.4, 0.5, 0.6],
}


def apply(config: dict, path: str, value) -> dict:
    patched = copy.deepcopy(config)
    target = patched
    *parents, leaf = path.split(".")
    for key in parents:
        target = target[key]
    target[leaf] = value
    return patched


def main() -> int:
    parser = argparse.ArgumentParser(description="Чувствительность ролей к порогам")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--out", type=Path, default=Path("out/sensitivity.csv"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = load_config(args.config)
    nodes, edges, _, _ = load_data(args.data, config)
    graph = build_graph(nodes, edges)
    features, _ = compute_features(graph, nodes, config)
    baseline = assign_roles(features, config).role

    rows = []
    for path, values in SWEEPS.items():
        current = config
        for key in path.split("."):
            current = current[key]
        for value in values:
            roles = assign_roles(features, apply(config, path, value)).role
            counts = roles.value_counts()
            rows.append({"порог": path, "значение": value, "текущий": value == current,
                         "изменилось_ролей": int((roles != baseline).sum()),
                         **{role: int(counts.get(role, 0)) for role in
                            ["consolidator", "distributor", "transit", "terminal", "coordinator", "peripheral"]}})

    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False, encoding="utf-8", lineterminator="\n")
    print(table.to_string(index=False))
    print(f"\nСохранено: {args.out}")
    print(json.dumps({"базовое_распределение": baseline.value_counts().to_dict()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
