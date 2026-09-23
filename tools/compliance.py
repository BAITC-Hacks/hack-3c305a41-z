"""Mechanical check against the brief, run before submitting.

The jury verifies the must-have list by opening the files and counting rows.
This does the same thing, so a broken requirement is found here and not there.

    python tools/compliance.py --out out
"""
import argparse
from pathlib import Path
import json
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

ROLES = {"consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral"}
NODE_COLUMNS = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]
CLUSTER_COLUMNS = ["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"]
TOP_COLUMNS = ["rank", "gid", "role", "priority_score", "why"]
README_SECTIONS = ["Запуск", "Как считается роль", "Ограничения", "Масштабирование", "Схема"]
# Section 9 of the brief: no wording that states guilt instead of a hypothesis.
FORBIDDEN = ["отмыва", "преступник", "виновен", "виновн", "мошенник", "причастен"]
# The brief also asks us to say what a score is not. Those denials are removed
# before the search, otherwise the required disclaimer trips its own check.
DISCLAIMERS = [r"не\s+оценка\s+виновности", r"не\s+вывод\s+о\s+безвредности",
               r"не\s+является\s+[^.]*виновност\w*", r"не\s+утвержд\w*\s+о\s+виновност\w*"]


def check(label: str, passed: bool, detail: str = "") -> bool:
    print(f"  [{'OK ' if passed else 'НЕТ'}] {label}{f' — {detail}' if detail else ''}")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка решения против требований ТЗ")
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    nodes = pd.read_csv(args.out / "nodes_roles.csv")
    clusters = pd.read_csv(args.out / "clusters.csv")
    top = pd.read_csv(args.out / "top_nodes.csv")
    manifest = json.loads((args.out / "run_manifest.json").read_text(encoding="utf-8"))
    results = []

    print("\nMUST HAVE 1 — воспроизводимый пайплайн")
    results.append(check("Полный расчёт укладывается в 300 с", manifest["within_time_limit"],
                         f"{manifest['total_seconds']:.2f} с"))
    results.append(check("Прогон не обращался к внешним сервисам", not manifest["llm_used"]))
    results.append(check("Хеши входов и артефактов записаны",
                         bool(manifest.get("input_sha256")) and bool(manifest.get("artifacts"))))

    print("\nMUST HAVE 2 — роль и скор у каждого узла")
    results.append(check("nodes_roles.csv содержит 2 248 строк", len(nodes) == 2248, str(len(nodes))))
    results.append(check("Обязательные колонки заполнены", not nodes[NODE_COLUMNS].isna().any().any()))
    results.append(check("Все роли из словаря ТЗ", set(nodes.role) <= ROLES, ", ".join(sorted(set(nodes.role)))))
    results.append(check("evidence непустой и не длиннее 200 символов",
                         bool(nodes.evidence.str.len().between(1, 200).all())))
    results.append(check("Скоры в диапазоне 0–1",
                         bool(nodes[["role_score", "priority_score"]].apply(lambda s: s.between(0, 1)).all().all())))

    print("\nMUST HAVE 3 — критерии задокументированы и объяснимы")
    readme = (args.root / "README.md").read_text(encoding="utf-8")
    results.append(check("Разделы README на месте",
                         all(section in readme for section in README_SECTIONS)))
    results.append(check("Пороги вынесены в config.yaml", (args.root / "config.yaml").is_file()))
    sources = "\n".join(path.read_text(encoding="utf-8") for path in (args.root / "src").rglob("*.py"))
    # The brief forbids baking a list of gids into the code instead of computing it.
    hardcoded = re.findall(r"\b1000000\d{11}\b", sources)
    results.append(check("В коде нет зашитых gid", not hardcoded, ", ".join(hardcoded[:3])))

    print("\nMUST HAVE 4 — кластеризация")
    sizes = nodes.groupby("cluster_id").size().sort_index()
    results.append(check("cluster_id есть у каждого узла", not nodes.cluster_id.isna().any()))
    results.append(check("Схема clusters.csv полная", set(CLUSTER_COLUMNS) <= set(clusters.columns)))
    results.append(check("Размеры кластеров согласованы",
                         clusters.set_index("cluster_id").n_nodes.sort_index().equals(sizes.rename("n_nodes"))))
    results.append(check("У каждого кластера есть гипотеза", bool(clusters.hypothesis.str.len().gt(0).all())))

    print("\nMUST HAVE 5 — топ-лист и визуализация")
    results.append(check("top_nodes.csv содержит не менее 20 строк", len(top) >= 20, str(len(top))))
    results.append(check("Схема top_nodes.csv полная", set(TOP_COLUMNS) <= set(top.columns)))
    results.append(check("Список отсортирован по приоритету", bool(top.priority_score.is_monotonic_decreasing)))
    results.append(check("Экран просмотра присутствует", (args.root / "app.py").is_file()))

    print("\nРАЗДЕЛ 9 — осторожность формулировок")
    texts = pd.concat([nodes.evidence, nodes.why, clusters.hypothesis, top.why]).str.lower()
    disclaimed = texts.replace("|".join(DISCLAIMERS), "", regex=True)
    found = sorted({word for word in FORBIDDEN if disclaimed.str.contains(word).any()})
    results.append(check("Нет утверждений о виновности в текстах выгрузок", not found, ", ".join(found)))
    results.append(check("Оговорка о характере скора присутствует",
                         bool(texts.str.contains("не оценка виновности").any())))

    passed = sum(results)
    print(f"\nИтог: {passed} из {len(results)} проверок пройдено.")
    if passed < len(results):
        print("Есть невыполненные требования — смотрите строки с пометкой НЕТ.")
        return 1
    print("Решение соответствует обязательным требованиям ТЗ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
