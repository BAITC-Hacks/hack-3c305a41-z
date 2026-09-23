"""Build a static copy of the report for GitHub Pages.

The Streamlit viewer needs Python running, which a static host cannot do. This
renders the same finished run into one offline page, so anyone with the link
sees the network, the ranking and the resilience check without installing
anything. It reads the published CSVs; nothing is recomputed here.

    python tools/build_site.py --out out --site docs
"""
import argparse
import html
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.narrate import ROLE_COLORS, ROLE_LABELS
from src.ui import STYLE, chip, eyebrow, key_values, money, priority_bar
from src.viewer import graph_html

PAGE_CSS = """<style>
body{margin:0;background:var(--bg);color:var(--text);
font-family:ui-sans-serif,system-ui,'Segoe UI',Roboto,Arial,sans-serif;line-height:1.55}
.wrap{max-width:1180px;margin:0 auto;padding:44px 22px 80px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:13px;margin:22px 0}
.stat{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:16px 18px}
.stat b{display:block;font-size:1.85rem;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.stat small{color:var(--muted);font-size:.79rem;display:block;margin-top:5px}
section{margin:38px 0}
h2{font-size:1.35rem;margin:0 0 6px}
table{width:100%;border-collapse:collapse;font-size:.88rem}
th,td{text-align:left;padding:9px 11px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-size:.72rem;letter-spacing:.09em;text-transform:uppercase;font-weight:700}
td.num{text-align:right;font-variant-numeric:tabular-nums}
iframe{width:100%;border:1px solid var(--line);border-radius:14px;background:#0e1726}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.links a{display:inline-block;margin:0 12px 10px 0;padding:9px 15px;border-radius:10px;
background:var(--raised);border:1px solid var(--line);color:var(--accent);
text-decoration:none;font-size:.87rem;font-weight:600}
.links a:hover{border-color:var(--accent)}
footer{color:var(--muted);font-size:.8rem;border-top:1px solid var(--line);padding-top:18px;margin-top:44px}
@media(max-width:860px){.cols{grid-template-columns:1fr}}
</style>"""


def svg_chart(table: pd.DataFrame, column: str, caption: str) -> str:
    """Two polylines, drawn by hand so the page needs no chart library."""
    width, height, pad = 520, 210, 34
    xs = sorted(table.removed.unique())
    top = max(table[column].max(), 1)
    paths = []
    for strategy, colour in (("priority", "#36d6bd"), ("random", "#8fa3bf")):
        series = table[table.strategy == strategy].sort_values("removed")
        points = " ".join(
            f"{pad + (row.removed / max(xs)) * (width - 2 * pad):.1f},"
            f"{height - pad - (getattr(row, column) / top) * (height - 2 * pad):.1f}"
            for row in series.itertuples())
        paths.append(f'<polyline fill="none" stroke="{colour}" stroke-width="2.5" points="{points}"/>')
    ticks = "".join(
        f'<text x="{pad + (x / max(xs)) * (width - 2 * pad):.0f}" y="{height - 10}" '
        f'fill="#8fa3bf" font-size="11" text-anchor="middle">{x}</text>' for x in xs)
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(caption)}" '
            f'style="width:100%;height:auto">'
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="#131d2f" rx="12"/>'
            f'<text x="{pad}" y="22" fill="#e6edf7" font-size="12" font-weight="600">{html.escape(caption)}</text>'
            f'<text x="{pad}" y="38" fill="#36d6bd" font-size="11">по приоритету</text>'
            f'<text x="{pad + 96}" y="38" fill="#8fa3bf" font-size="11">случайно</text>'
            f'{"".join(paths)}{ticks}</svg>')


def build(out: Path, site: Path, repo: str) -> None:
    nodes = pd.read_parquet(out / "node_features.parquet")
    edges = pd.read_parquet(out / "viewer_edges.parquet")
    clusters = pd.read_csv(out / "clusters.csv")
    top = pd.read_csv(out / "top_nodes.csv")
    resilience = pd.read_csv(out / "resilience.csv")
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    config, summary = manifest["config"], manifest.get("resilience", {})

    site.mkdir(parents=True, exist_ok=True)
    (site / "data").mkdir(exist_ok=True)
    for name in ("nodes_roles.csv", "clusters.csv", "top_nodes.csv", "resilience.csv"):
        shutil.copy(out / name, site / "data" / name)

    shown = nodes.sort_values(["priority_score", "gid"], ascending=[False, True]).head(config["viewer"]["max_nodes"])
    links = edges[edges.src.isin(shown.gid) & edges.dst.isin(shown.gid)]
    (site / "graph.html").write_text(graph_html(shown, links, config), encoding="utf-8")

    stats = [("Участники", money(len(nodes)), "включая изолированных; исходный список — 81 клиент"),
             ("Связи", money(len(edges)), "пары плательщик → получатель за июль 2026"),
             ("Точки сбора", money(int(nodes.role.isin(["consolidator", "coordinator"]).sum())),
              "роли «Консолидация» и «Координация»"),
             ("Граница наблюдения", money(int(nodes.truncated_by_depth.sum())),
              "узлы на краю обхода: не «деньги осели», а конец выгрузки"),
             ("Полный расчёт", f"{manifest['total_seconds']:.1f} с", "при ограничении ТЗ в 300 с")]
    cards = "".join(f'<div class="stat">{eyebrow(name)}<b>{value}</b><small>{note}</small></div>'
                    for name, value, note in stats)

    leaders = "".join(
        f'<div class="gm-card">{eyebrow(f"Место {int(row["rank"])}")}'
        f'<div class="gm-id" style="font-size:1.1rem">gid {int(row.gid)}</div>{chip(row.role)}'
        f'<div style="margin-top:10px">{priority_bar(row)}</div>'
        f'<div class="gm-quote" style="margin-top:12px;font-size:.86rem">{html.escape(row.evidence)}</div></div>'
        for _, row in nodes.nsmallest(3, "rank").iterrows())

    rows = "".join(
        f'<tr><td class="num">{int(r.rank)}</td><td>{int(r.gid)}</td>'
        f'<td>{chip(nodes.loc[nodes.gid == r.gid, "role"].iloc[0])}</td>'
        f'<td class="num">{r.priority_score:.3f}</td><td>{html.escape(str(r.why))}</td></tr>'
        for r in top.itertuples())

    distribution = "".join(
        f'<div style="margin-bottom:9px">{chip(role)}'
        f'<span style="float:right;font-variant-numeric:tabular-nums;font-weight:600">'
        f'{money(int((nodes.role == role).sum()))}</span>'
        f'<div class="gm-bar" style="margin:6px 0 0"><span style="width:'
        f'{(nodes.role == role).mean() * 100:.1f}%;background:{ROLE_COLORS[role]}"></span></div></div>'
        for role in ROLE_LABELS)

    verdict = ""
    if summary.get("priority_largest_component"):
        verdict = (f'<div class="gm-card">{eyebrow("Что это значит")}'
                   f'<p class="gm-lede">Изъятие {summary["headline_step"]} участников из верха списка отрезает '
                   f'от ядра сети на <b>{summary["advantage_nodes_detached"]}</b> участников больше, чем изъятие '
                   f'такого же числа случайных, и оставляет в ядре {summary["priority_turnover_share"]:.0%} '
                   f'оборота вместо {summary["random_turnover_share"]:.0%}. Если бы порядок в списке был '
                   'неинформативным, обе кривые совпали бы.</p></div>')

    page = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Граф денег — отчёт по сети переводов</title>
<meta name="description" content="Роли участников транзакционной сети и приоритет проверки для AML-аналитика.">
{STYLE}{PAGE_CSS}</head><body><div class="wrap">
{eyebrow("HackAlem AI · трек «Финансы» · июль 2026")}
<h1 style="font-size:2.3rem;margin:0 0 10px">Граф денег</h1>
<p class="gm-lede" style="max-width:760px">AML-аналитику известен 81 клиент — нижний уровень цепочки.
Наблюдаемая сеть их переводов на четыре колена — {money(len(nodes))} участников. Инструмент строит граф,
присваивает каждому роль по документированным правилам с порогами и отвечает, кого проверять первым
и какие числа это объясняют.</p>
<p class="gm-lede" style="max-width:760px;margin-top:10px">Это статическая копия готового прогона.
Роли — гипотезы для проверки, а не утверждение о виновности; скоры не являются вероятностью нарушения.</p>

<div class="grid">{cards}</div>

<section><h2>Кого проверять первым</h2>
<p class="gm-lede">Приоритет складывается из четырёх слагаемых: вес роли, связь с исходным списком,
объём и посредничество. Каждое видно отдельно.</p>
<div class="cols" style="grid-template-columns:repeat(auto-fit,minmax(260px,1fr))">{leaders}</div>
<table><thead><tr><th>Место</th><th>gid</th><th>Роль</th><th>Приоритет</th><th>Обоснование</th></tr></thead>
<tbody>{rows}</tbody></table></section>

<section><h2>Схема сети</h2>
<p class="gm-lede">{len(shown)} участников с наибольшим приоритетом и связи между ними. Цвет — роль,
стрелка — направление перевода, толщина — сумма. Узлы можно перетаскивать.</p>
<iframe src="graph.html" height="{config["viewer"]["height"] + 12}" title="Направленный граф переводов"></iframe></section>

<section><h2>Устойчивость сети: проверка самого рейтинга</h2>
<p class="gm-lede">Узлы изымаются двумя способами — по нашему приоритету и случайно, случайный вариант
усреднён по нескольким розыгрышам с фиксированными сидами.</p>
{verdict}
<div class="cols">{svg_chart(resilience, "largest_component", "Крупнейший связный фрагмент")}
{svg_chart(resilience, "components_min2", "Число фрагментов сети")}</div>
<p class="gm-lede" style="margin-top:8px">По горизонтали — сколько участников изъято. Оценка описывает
наблюдаемую сеть и не является прогнозом последствий каких-либо действий в отношении клиентов.</p></section>

<section><h2>Роли и группы</h2><div class="cols">
<div class="gm-card">{eyebrow("Распределение ролей")}{distribution}</div>
<div class="gm-card">{eyebrow("Кластеризация")}
{key_values([("Групп всего", len(clusters)), ("С более чем одним seed", int((clusters.n_seed > 1).sum())),
             ("Крупнейшая группа", money(int(clusters.n_nodes.max()))),
             ("Участников вне крупнейшей компоненты", money(manifest["diagnostics"]["outside_largest_component"]))])}
<p class="gm-lede" style="margin-top:10px">Louvain на неориентированной проекции; роли и направление
потоков рассчитаны на направленном графе.</p></div></div></section>

<section><h2>Проверить и запустить</h2>
<p class="gm-lede">Полный инструмент работает локально и не требует ключей: один запуск от сырых
parquet до трёх выгрузок за {manifest["total_seconds"]:.1f} с.</p>
<div class="links"><a href="{repo}">Репозиторий и README</a>
<a href="data/nodes_roles.csv">nodes_roles.csv</a><a href="data/top_nodes.csv">top_nodes.csv</a>
<a href="data/clusters.csv">clusters.csv</a><a href="data/resilience.csv">resilience.csv</a></div></section>

<footer>Данные обезличены: gid — синтетический идентификатор без привязки к личности.
Страница собрана из опубликованного прогона от {manifest["created_at_utc"][:10]}
командой <code>python tools/build_site.py</code>.</footer>
</div></body></html>"""
    (site / "index.html").write_text(page, encoding="utf-8")
    (site / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Собрано: {(site / 'index.html').resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Статическая копия отчёта для GitHub Pages")
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--site", type=Path, default=Path("docs"))
    parser.add_argument("--repo", default="https://github.com/BAITC-Hacks/hack-3c305a41-z")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    build(args.out, args.site, args.repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
