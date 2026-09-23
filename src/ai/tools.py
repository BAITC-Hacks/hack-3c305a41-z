"""Graph lookups the assistant may call. The model reads facts, it never computes them."""
import json
from typing import Any

import networkx as nx
import pandas as pd

from src.ai import agents as roster
from src.ai.client import LLMUnavailable, build_client, create_chat

SCHEMA = [
    {"type": "function", "function": {"name": "node_card", "description": "Карточка узла: роль, метрики, обоснование, ограничения данных.",
     "parameters": {"type": "object", "properties": {"gid": {"type": "integer"}}, "required": ["gid"]}}},
    {"type": "function", "function": {"name": "neighbours", "description": "Контрагенты узла с суммами переводов.",
     "parameters": {"type": "object", "properties": {"gid": {"type": "integer"}, "direction": {"type": "string", "enum": ["in", "out"]},
                                                     "limit": {"type": "integer"}}, "required": ["gid", "direction"]}}},
    {"type": "function", "function": {"name": "path_between", "description": "Кратчайший путь движения денег между двумя узлами.",
     "parameters": {"type": "object", "properties": {"src": {"type": "integer"}, "dst": {"type": "integer"}}, "required": ["src", "dst"]}}},
    {"type": "function", "function": {"name": "top_by_role", "description": "Узлы заданной роли по убыванию приоритета проверки.",
     "parameters": {"type": "object", "properties": {"role": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["role"]}}},
    {"type": "function", "function": {"name": "collectors_for", "description": "Кому перечисляют деньги указанные узлы: общие получатели в пределах двух шагов.",
     "parameters": {"type": "object", "properties": {"gids": {"type": "array", "items": {"type": "integer"}}}, "required": ["gids"]}}},
]

CARD_FIELDS = ["gid", "role", "role_score", "cluster_id", "priority_score", "rank", "in_deg", "out_deg",
               "in_kzt", "out_kzt", "pass_through", "n_seed_sources", "depth", "is_seed",
               "truncated_by_depth", "evidence", "data_limitation", "next_request"]


def _frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", double_precision=6))


class GraphTools:
    """Bound to one finished run: the answers always match the published CSV."""

    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
        self.nodes = nodes.set_index("gid", drop=False)
        self.edges = edges
        self.graph = nx.from_pandas_edgelist(edges, "src", "dst", edge_attr=True, create_using=nx.DiGraph)

    def node_card(self, gid: int) -> dict[str, Any]:
        if gid not in self.nodes.index:
            return {"error": f"gid {gid} отсутствует в выгрузке"}
        row = self.nodes.loc[gid, [c for c in CARD_FIELDS if c in self.nodes.columns]]
        return _frame_to_records(row.to_frame().T)[0]

    def neighbours(self, gid: int, direction: str, limit: int = 10) -> list[dict[str, Any]]:
        column, other = ("dst", "src") if direction == "out" else ("src", "dst")
        subset = self.edges[self.edges[other] == gid].nlargest(limit, "sum_kzt")
        subset = subset.assign(role=subset[column].map(self.nodes.role))
        return _frame_to_records(subset[[column, "sum_kzt", "n_tx", "role"]])

    def path_between(self, src: int, dst: int) -> dict[str, Any]:
        if src not in self.graph or dst not in self.graph:
            return {"error": "Один из узлов отсутствует в наблюдаемых рёбрах"}
        try:
            path = nx.shortest_path(self.graph, src, dst)
        except nx.NetworkXNoPath:
            return {"path": [], "note": "Путь в наблюдаемых данных не найден; возможен перевод вне выборки"}
        return {"path": path, "steps": [{"src": a, "dst": b, "sum_kzt": self.graph[a][b]["sum_kzt"]} for a, b in zip(path, path[1:])]}

    def top_by_role(self, role: str, limit: int = 10) -> list[dict[str, Any]]:
        subset = self.nodes[self.nodes.role == role].nsmallest(limit, "rank")
        return _frame_to_records(subset[["gid", "role", "priority_score", "rank", "evidence"]])

    def collectors_for(self, gids: list[int]) -> list[dict[str, Any]]:
        counts: dict[int, set[int]] = {}
        for gid in gids:
            if gid not in self.graph:
                continue
            for target in nx.single_source_shortest_path_length(self.graph, gid, cutoff=2):
                if target != gid:
                    counts.setdefault(target, set()).add(gid)
        shared = sorted(counts.items(), key=lambda item: (-len(item[1]), item[0]))[:10]
        return [{"gid": gid, "получает_от": sorted(sources), "роль": self.nodes.role.get(gid, "нет данных"),
                 "приоритет": float(self.nodes.priority_score.get(gid, 0))} for gid, sources in shared if len(sources) > 1]

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        return getattr(self, name)(**arguments)


def answer(question: str, tools: GraphTools, config: dict[str, Any]) -> dict[str, Any]:
    """Tool-calling loop. Returns the answer plus the exact calls made, so it is auditable."""
    settings = config["llm"]
    client = build_client(settings)
    messages = [{"role": "system", "content": roster.ASSISTANT.system}, {"role": "user", "content": question}]
    trace: list[dict[str, Any]] = []
    for _ in range(settings["assistant_max_steps"]):
        response = create_chat(client, settings, settings["models"]["assistant"], messages, tools=SCHEMA)
        message = response.choices[0].message
        if not message.tool_calls:
            content = json.loads(message.content) if message.content and message.content.strip().startswith("{") else {"answer": message.content}
            return {"answer": content.get("answer", ""), "gids": content.get("gids", []), "trace": trace}
        messages.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls:
            arguments = json.loads(call.function.arguments or "{}")
            result = tools.call(call.function.name, arguments)
            trace.append({"tool": call.function.name, "arguments": arguments})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False, default=str)})
    raise LLMUnavailable("Ассистент не уложился в лимит шагов")
