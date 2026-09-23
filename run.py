"""Single local entry point: raw parquet -> validated CSV and viewer bundle."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter

from src.clusters import assign_clusters, summarize_clusters
from src.config import load_config
from src.export import file_hash, validate_outputs, write_outputs
from src.features import compute_features
from src.gaps import add_gaps
from src.graph import build_graph
from src.load import load_data
from src.narrate import add_evidence
from src.priority import rank_nodes
from src.resilience import fragmentation
from src.roles import assign_roles
from src.temporal import compute_temporal, summarize


def run(data: Path, out: Path, config_path: Path, llm: bool = False) -> dict:
    start = perf_counter()
    config = load_config(config_path)
    if out.resolve() == data.resolve():
        raise ValueError("Папки данных и результатов должны различаться")
    timings = {}

    def timed(label, operation):
        tick = perf_counter()
        result = operation()
        timings[label] = perf_counter() - tick
        print(f"{label}: {timings[label]:.3f} с")
        return result

    nodes, edges, tx, data_report = timed("Проверка входа", lambda: load_data(data, config))
    graph = timed("Граф", lambda: build_graph(nodes, edges))
    features, diagnostics = timed("Метрики", lambda: compute_features(graph, nodes, config))
    features = timed("Время", lambda: features.merge(compute_temporal(tx, nodes, config), on="gid", how="left"))
    roles = timed("Роли", lambda: add_evidence(add_gaps(assign_roles(features, config)), config))
    roles = timed("Кластеры", lambda: assign_clusters(roles, graph, config))
    roles, top = timed("Приоритет", lambda: rank_nodes(roles, config))
    clusters = summarize_clusters(roles, edges, config)
    resilience, resilience_summary = timed("Устойчивость", lambda: fragmentation(graph, roles, edges, config))
    narrative = None
    if llm:
        # Optional layer: a failure here must never cost us the deterministic artifacts.
        from src.ai.client import LLMUnavailable
        from src.ai.orchestrator import enrich
        try:
            roles, clusters, top, narrative = timed("ИИ-слой", lambda: enrich(roles, clusters, top, config))
        except LLMUnavailable as error:
            print(f"ИИ-слой пропущен, тексты остаются офлайн-расчётом: {error}")
    validate_outputs(roles, clusters, top, nodes, config)
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                "config_sha256": file_hash(config_path), "input_sha256": {p.name: file_hash(p) for p in sorted(data.glob("*.parquet"))},
                "python": sys.version.split()[0], "packages": {name: version(name) for name in ("pandas", "numpy", "networkx", "pyarrow", "scipy", "PyYAML")},
                "data_report": data_report, "diagnostics": diagnostics, "role_counts": roles.role.value_counts().sort_index().to_dict(),
                "cluster_count": len(clusters), "clusters_with_multiple_seeds": int((clusters.n_seed > 1).sum()),
                "temporal": summarize(roles), "resilience": resilience_summary,
                "timings_seconds": timings, "llm_used": narrative is not None,
                "llm": narrative}
    timed("Экспорт", lambda: write_outputs(roles, clusters, top, edges, resilience, out, manifest))
    if narrative is not None:
        (out / "llm_review.json").write_text(json.dumps(narrative, ensure_ascii=False, indent=2), encoding="utf-8")
    elapsed = perf_counter() - start
    manifest["total_seconds"] = elapsed
    manifest["within_time_limit"] = elapsed <= config["export"]["max_runtime_seconds"]
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"данные": data_report, "граф": diagnostics, "роли": manifest["role_counts"], "кластеры": len(clusters)}, ensure_ascii=False, indent=2))
    print(f"Готово: {out.resolve()} — {elapsed:.3f} с")
    if not manifest["within_time_limit"]:
        raise ValueError("Превышен лимит полного расчёта; результаты сохранены, нужна оптимизация")
    return manifest


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Объяснимый анализ графа переводов")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--llm", action="store_true", help="Слой агентов: переписывает тексты поверх готовых чисел")
    args = parser.parse_args()
    try:
        run(args.data, args.out, args.config, args.llm)
    except (ValueError, OSError, KeyError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
