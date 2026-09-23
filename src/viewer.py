"""Read one verified run and render directed graphs with bundled JS only."""
import html
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pyvis
from pyvis.network import Network

from src.export import file_hash
from src.narrate import ROLE_COLORS, ROLE_LABELS


def read_bundle(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest_path = folder / "run_manifest.json"
    if not manifest_path.exists():
        raise ValueError("Результаты не найдены. Сначала выполните python run.py --data data --out out")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"nodes_roles.csv", "clusters.csv", "top_nodes.csv", "node_features.parquet", "viewer_edges.parquet"}
    if set(manifest.get("artifacts", {})) != required:
        raise ValueError("Неполный набор результатов. Повторите расчёт.")
    for name, expected in manifest["artifacts"].items():
        path = folder / name
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Результаты изменены или смешаны: {name}. Повторите расчёт.")
    return (pd.read_parquet(folder / "node_features.parquet"), pd.read_parquet(folder / "viewer_edges.parquet"),
            pd.read_csv(folder / "clusters.csv"), pd.read_csv(folder / "top_nodes.csv"), manifest)


def select_neighborhood(nodes: pd.DataFrame, edges: pd.DataFrame, gid: int, max_nodes: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    incident = edges[(edges.src == gid) | (edges.dst == gid)].copy()
    neighbors = set(incident.src) | set(incident.dst) | {gid}
    ranked = nodes[nodes.gid.isin(neighbors - {gid})].sort_values(["priority_score", "gid"], ascending=[False, True])
    selected = {gid} | set(ranked.head(max_nodes - 1).gid)
    shown_nodes = nodes[nodes.gid.isin(selected)].copy()
    shown_edges = edges[edges.src.isin(selected) & edges.dst.isin(selected)].copy()
    return shown_nodes, shown_edges, incident


def graph_html(nodes: pd.DataFrame, edges: pd.DataFrame, config: dict[str, Any], selected_gid: int | None = None) -> str:
    height = config["viewer"]["height"]
    network = Network(height=f"{height}px", width="100%", directed=True, cdn_resources="in_line")
    for row in nodes.itertuples(index=False):
        title = html.escape(f"gid {row.gid} | {ROLE_LABELS[row.role]} | кластер {row.cluster_id}\n{row.evidence}")
        network.add_node(str(row.gid), label=str(row.gid), title=title, color=ROLE_COLORS[row.role],
                         shape="diamond" if row.is_seed else "dot", size=12 + 18 * row.priority_score,
                         borderWidth=4 if row.gid == selected_gid else 1,
                         font={"color": "#e2e8f0", "size": 12})
    for row in edges.itertuples(index=False):
        network.add_edge(str(row.src), str(row.dst), arrows="to", width=1 + math.log1p(row.sum_kzt) / 6,
                         title=html.escape(f"{row.src} → {row.dst}: {row.sum_kzt:,.2f} KZT, переводов {row.n_tx}"))
    options = {"layout": {"randomSeed": config["viewer"]["layout_seed"]},
               "interaction": {"hover": True, "navigationButtons": False},
               "edges": {"color": {"color": "#526882", "highlight": "#f8fafc"}, "smooth": {"type": "dynamic"}},
               "physics": {"stabilization": {"iterations": config["viewer"]["stabilization_iterations"]}}}
    resources = Path(pyvis.__file__).parent / "templates" / "lib" / "vis-9.1.2"
    js = (resources / "vis-network.min.js").read_text(encoding="utf-8")
    css = (resources / "vis-network.css").read_text(encoding="utf-8")
    # Avoid pyvis' default template: it adds remote Bootstrap even in in_line mode.
    data = json.dumps({"nodes": network.nodes, "edges": network.edges}, ensure_ascii=False).replace("</", "<\\/")
    return (f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><style>{css}\n'
            f'html,body{{margin:0;background:#0e1726}}#graph{{height:{height}px;width:100%}}</style></head>'
            f'<body><div id="graph" role="img" aria-label="Направленный граф переводов"></div><script>{js}</script>'
            f'<script>const graphData={data};const graphOptions={json.dumps(options)};'
            'window.network=new vis.Network(document.getElementById("graph"),graphData,graphOptions);'
            'network.once("stabilizationIterationsDone",()=>network.setOptions({physics:false}));'
            '</script></body></html>')
