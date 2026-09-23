"""Configuration loading and validation."""
from pathlib import Path
import math
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Конфигурация должна быть словарём YAML")
    for section in ("data", "features", "roles", "clusters", "priority", "export", "viewer", "temporal", "resilience"):
        if section not in config:
            raise ValueError(f"Отсутствует раздел конфигурации: {section}")
    role_names = {"coordinator", "consolidator", "distributor", "transit", "terminal"}
    order = config["roles"]["tie_order"]
    if set(order) != role_names or len(order) != len(role_names):
        raise ValueError("tie_order должен содержать пять разных специализированных ролей")
    groups = [config["priority"]["weights"]] + [config["roles"][r]["weights"] for r in order]
    for weights in groups:
        if not all(math.isfinite(v) and v >= 0 for v in weights.values()) or not math.isclose(sum(weights.values()), 1):
            raise ValueError("Веса должны быть неотрицательными, конечными и давать сумму 1")
    for value in [config["roles"]["minimum_score"], config["roles"]["fallback_cap"], *config["priority"]["role_values"].values()]:
        if not 0 <= value <= 1:
            raise ValueError("Скоры и веса ролей должны находиться в 0–1")
    if not 0 < config["features"]["normalization_quantile"] <= 1:
        raise ValueError("Квантиль нормировки должен находиться в (0, 1]")
    transit = config["roles"]["transit"]
    if not 0 <= transit["min_ratio"] < 1 < transit["max_ratio"]:
        raise ValueError("Границы transit должны охватывать единицу")
    steps = config["resilience"]["steps"]
    if sorted(set(steps)) != list(steps) or steps[0] != 0:
        raise ValueError("resilience.steps должен возрастать и начинаться с нуля")
    if config["resilience"]["headline_step"] not in steps:
        raise ValueError("resilience.headline_step должен быть одним из steps")
    if not 0 < config["temporal"]["fast_pass_min_share"] <= 1:
        raise ValueError("temporal.fast_pass_min_share должен находиться в (0, 1]")
    for section, key in (("priority", "top_count"), ("viewer", "max_nodes"), ("clusters", "top_gids_count")):
        if not isinstance(config[section][key], int) or config[section][key] <= 0:
            raise ValueError(f"{section}.{key} должен быть положительным целым")
    return config
