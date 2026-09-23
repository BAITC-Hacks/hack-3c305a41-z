from copy import deepcopy
from pathlib import Path
from html.parser import HTMLParser
import hashlib

import networkx as nx
import pandas as pd
import pytest

from run import run
from src.config import load_config
from src.features import compute_features
from src.graph import build_graph, community_projection
from src.load import load_data
from src.narrate import add_evidence
from src.roles import assign_roles
from src.viewer import graph_html, read_bundle, select_neighborhood

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return load_config(ROOT / "config.yaml")


def small_input(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Explicitly synthetic chain with an isolate and a truncated endpoint."""
    folder.mkdir(parents=True, exist_ok=True)
    nodes = pd.DataFrame({"gid": [1, 2, 3, 4, 5], "depth": [0, 1, 2, 4, 0], "is_seed": [True, False, False, False, True]})
    tx = pd.DataFrame({"src": [1, 2, 3], "dst": [2, 3, 4], "date": pd.to_datetime(["2026-07-01"] * 3), "sum_kzt": [100000.0] * 3})
    edges = tx.drop(columns="date").assign(n_tx=1, depth=[1, 2, 4])
    for name, table in (("nodes", nodes), ("edges", edges), ("transactions", tx)):
        table.to_parquet(folder / f"{name}.parquet", index=False)
    return nodes, edges, tx


def test_loader_detects_amount_mismatch(tmp_path, config):
    _, edges, _ = small_input(tmp_path)
    edges.loc[0, "sum_kzt"] += 100
    edges.to_parquet(tmp_path / "edges.parquet", index=False)
    with pytest.raises(ValueError, match="Суммы"):
        load_data(tmp_path, config)


def test_loader_detects_count_mismatch(tmp_path, config):
    _, edges, _ = small_input(tmp_path)
    edges.loc[0, "n_tx"] += 1
    edges.to_parquet(tmp_path / "edges.parquet", index=False)
    with pytest.raises(ValueError, match="Количество"):
        load_data(tmp_path, config)


def test_loader_rejects_unknown_node(tmp_path, config):
    _, edges, _ = small_input(tmp_path)
    edges.loc[0, "dst"] = 99
    edges.to_parquet(tmp_path / "edges.parquet", index=False)
    with pytest.raises(ValueError, match="отсутствует в nodes"):
        load_data(tmp_path, config)


def test_loader_rejects_duplicate_gid(tmp_path, config):
    nodes, _, _ = small_input(tmp_path)
    pd.concat([nodes, nodes.iloc[[0]]]).to_parquet(tmp_path / "nodes.parquet", index=False)
    with pytest.raises(ValueError, match="Дубликат"):
        load_data(tmp_path, config)


def test_loader_missing_file(tmp_path, config):
    with pytest.raises(ValueError, match="Нет входного файла"):
        load_data(tmp_path, config)


def test_chain_and_isolate_roles(tmp_path, config):
    nodes, edges, _ = small_input(tmp_path)
    graph = build_graph(nodes, edges)
    features, report = compute_features(graph, nodes, config)
    result = add_evidence(assign_roles(features, config), config).set_index("gid")
    assert len(graph) == len(nodes)
    assert result.loc[2, "role"] == "transit"
    assert result.loc[3, "role"] == "transit"
    assert result.loc[4, "role"] == "peripheral"
    assert result.loc[5, "role_score"] == 0
    assert pd.isna(result.loc[1, "pass_through"])
    assert not result.loc[1, "eligible_transit"]
    assert report["weak_components_with_isolates"] == 2
    assert report["isolated_nodes"] == 1
    assert result.evidence.str.len().max() <= config["export"]["evidence_max_chars"]


def test_fan_in_and_fan_out(config):
    graph = nx.DiGraph()
    graph.add_edges_from([(i, 20) for i in range(1, 6)] + [(30, i) for i in range(40, 52)])
    nx.set_edge_attributes(graph, 100000.0, "sum_kzt")
    nx.set_edge_attributes(graph, 1, "n_tx")
    nodes = pd.DataFrame({"gid": sorted(graph.nodes), "depth": 1, "is_seed": False})
    features, _ = compute_features(graph, nodes, config)
    result = assign_roles(features, config).set_index("gid")
    assert result.loc[20, "role"] == "consolidator"
    assert result.loc[30, "role"] == "distributor"
    assert result.loc[40, "role"] == "terminal"


def test_cycle_does_not_count_seed_as_its_own_source(config):
    nodes = pd.DataFrame({"gid": [1, 2, 3], "depth": [0, 1, 2], "is_seed": [True, False, False]})
    edges = pd.DataFrame({"src": [1, 2, 3], "dst": [2, 3, 1], "sum_kzt": [5000.0] * 3, "n_tx": [1] * 3})
    frame, _ = compute_features(build_graph(nodes, edges), nodes, config)
    assert frame.n_seed_sources.tolist() == [0, 1, 1]


def test_projection_sums_reciprocal_edges():
    graph = nx.DiGraph()
    graph.add_node(9)
    graph.add_edge(1, 2, sum_kzt=10)
    graph.add_edge(2, 1, sum_kzt=30)
    graph.add_edge(2, 2, sum_kzt=50)
    projected = community_projection(graph)
    assert projected[1][2]["weight"] == 40
    assert 9 in projected and not projected.has_edge(2, 2)


def test_seed_and_boundary_cannot_use_retention(tmp_path, config):
    nodes, edges, _ = small_input(tmp_path)
    frame, _ = compute_features(build_graph(nodes, edges), nodes, config)
    # Even strong incoming figures must not turn missing boundary outflows into retention.
    frame.loc[frame.gid.isin([1, 4]), ["in_deg", "in_kzt", "norm_in_deg", "norm_in_kzt", "in_concentration", "pass_through"]] = [20, 1000000, 1, 1, 0.05, 0]
    result = assign_roles(frame, config).set_index("gid")
    for gid in [1, 4]:
        assert not result.loc[gid, "eligible_terminal"]
        assert not result.loc[gid, "eligible_consolidator"]
        assert not result.loc[gid, "eligible_transit"]


def test_end_to_end_reproducibility_and_tamper(tmp_path):
    data = tmp_path / "data"
    small_input(data)
    first, second = tmp_path / "one", tmp_path / "two"
    run(data, first, ROOT / "config.yaml")
    # Reversing input rows must not change the result.
    for file in data.glob("*.parquet"):
        pd.read_parquet(file).iloc[::-1].to_parquet(file, index=False)
    run(data, second, ROOT / "config.yaml", llm=True)
    for name in ["nodes_roles.csv", "clusters.csv", "top_nodes.csv"]:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    nodes, edges, clusters, top, manifest = read_bundle(first)
    assert len(nodes) == clusters.n_nodes.sum() == 5
    assert len(top) == 5 and not manifest["llm_used"]
    assert nodes.priority_score.between(0, 1).all()
    with (first / "nodes_roles.csv").open("a") as handle:
        handle.write("tampered")
    with pytest.raises(ValueError, match="смешаны"):
        read_bundle(first)


def test_all_isolates_export(tmp_path):
    data = tmp_path / "data"
    nodes, edges, tx = small_input(data)
    edges.iloc[:0].to_parquet(data / "edges.parquet", index=False)
    tx.iloc[:0].to_parquet(data / "transactions.parquet", index=False)
    run(data, tmp_path / "out", ROOT / "config.yaml")
    result, _, clusters, _, _ = read_bundle(tmp_path / "out")
    assert len(clusters) == len(nodes)
    assert result.role.eq("peripheral").all() and result.role_score.eq(0).all()


def test_graph_is_offline_and_preserves_ids(tmp_path, config):
    data = tmp_path / "data"
    small_input(data)
    run(data, tmp_path / "out", ROOT / "config.yaml")
    nodes, edges, _, _, _ = read_bundle(tmp_path / "out")
    selected, links, incident = select_neighborhood(nodes, edges, 2, 2)
    assert 2 in selected.gid.values and len(selected) == 2
    assert len(incident) == 2
    rendered = graph_html(selected, links, config, 2)

    class Resources(HTMLParser):
        def handle_starttag(self, tag, attrs):
            assert not (tag in {"script", "link"} and any(k in {"src", "href"} and v.startswith("http") for k, v in attrs))

    Resources().feed(rendered)
    assert '"arrows": "to"' in rendered
    assert '"id": "2"' in rendered
    isolated, isolated_edges, _ = select_neighborhood(nodes, edges, 5, 2)
    assert isolated.gid.tolist() == [5] and isolated_edges.empty
