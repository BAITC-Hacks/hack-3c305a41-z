"""Graph construction preserving all declared nodes."""
import networkx as nx
import pandas as pd


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    for row in nodes.sort_values("gid").itertuples(index=False):
        graph.add_node(int(row.gid), depth=int(row.depth), is_seed=bool(row.is_seed))
    for row in edges.sort_values(["src", "dst"]).itertuples(index=False):
        graph.add_edge(int(row.src), int(row.dst), sum_kzt=float(row.sum_kzt), n_tx=int(row.n_tx))
    return graph


def community_projection(graph: nx.DiGraph) -> nx.Graph:
    projection = nx.Graph()
    projection.add_nodes_from(sorted(graph.nodes))
    for src, dst, data in sorted(graph.edges(data=True)):
        # A transfer to oneself does not connect two participants; keep it in turnover only.
        if src == dst:
            continue
        prior = projection.get_edge_data(src, dst, {}).get("weight", 0)
        projection.add_edge(src, dst, weight=prior + data["sum_kzt"])
    return projection
