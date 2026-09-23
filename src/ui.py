"""Presentation layer: one visual system, no remote resources.

Everything here is inline CSS and small HTML builders. The tool must run on an
analyst laptop with no internet, so no web fonts and no CDN are used.
"""
from typing import Any

import pandas as pd

from src.narrate import ROLE_COLORS, ROLE_LABELS

PRIORITY_PARTS = [
    ("priority_role", "Роль", "#36d6bd"),
    ("priority_sources", "Связь с seed", "#f59e0b"),
    ("priority_volume", "Объём", "#3b82f6"),
    ("priority_brokerage", "Посредничество", "#a78bfa"),
]

_ROLE_RULES = "\n".join(
    f'.gm-chip[data-role="{role}"]{{color:{color};background:{color}1f;border-color:{color}59}}'
    f'.gm-dot[data-role="{role}"]{{background:{color}}}'
    for role, color in ROLE_COLORS.items()
)

CSS = """<style>
:root{--bg:#0b1220;--surface:#131d2f;--raised:#18243a;--line:#223049;
--text:#e6edf7;--muted:#8fa3bf;--accent:#36d6bd;--warn:#f59e0b}

.block-container{max-width:1560px;padding-top:3.2rem;padding-bottom:4rem}
h1,h2,h3{letter-spacing:-.02em}
h1{font-size:2.35rem;font-weight:700;margin:0 0 .35rem}

/* Eyebrow: a quiet label above a block, so every section says what it is. */
.gm-eyebrow{font-size:.68rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;
color:var(--muted);margin:0 0 .55rem}
.gm-lede{color:var(--muted);font-size:.95rem;line-height:1.55;margin:0 0 .2rem}

.gm-card{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin-bottom:14px}
.gm-card--flat{background:transparent}

.gm-chip{display:inline-flex;align-items:center;gap:7px;padding:4px 11px;border-radius:999px;
border:1px solid;font-size:.78rem;font-weight:600;line-height:1.5;white-space:nowrap}
.gm-dot{width:8px;height:8px;border-radius:50%;display:inline-block;flex:none}

.gm-id{font-size:1.65rem;font-weight:700;letter-spacing:-.02em;margin:2px 0 8px;
font-variant-numeric:tabular-nums;word-break:break-all}
.gm-rank{display:inline-block;padding:3px 10px;border-radius:8px;background:var(--raised);
border:1px solid var(--line);font-size:.78rem;color:var(--muted);font-weight:600}

/* Stacked bar: the priority score split into the four numbers that produced it. */
.gm-bar{display:flex;height:11px;border-radius:6px;overflow:hidden;background:var(--raised);
border:1px solid var(--line);margin:10px 0 9px}
.gm-bar span{display:block;height:100%}
.gm-legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:.78rem;color:var(--muted)}
.gm-legend b{color:var(--text);font-weight:600;font-variant-numeric:tabular-nums}

.gm-kv{display:grid;grid-template-columns:auto 1fr;gap:6px 18px;font-size:.9rem;margin:2px 0}
.gm-kv dt{color:var(--muted)}
.gm-kv dd{margin:0;text-align:right;font-variant-numeric:tabular-nums;font-weight:600}

.gm-quote{border-left:3px solid var(--accent);background:var(--raised);border-radius:0 10px 10px 0;
padding:12px 15px;font-size:.92rem;line-height:1.6;margin:4px 0}
.gm-note{border-left:3px solid var(--warn);background:#f59e0b14;border-radius:0 10px 10px 0;
padding:11px 15px;font-size:.86rem;line-height:1.55;color:#fcd9a0;margin:10px 0 2px}
.gm-note b{color:#fde3ba}

.gm-pair{display:flex;gap:12px;flex-wrap:wrap}
.gm-pair>div{flex:1 1 150px;background:var(--raised);border:1px solid var(--line);
border-radius:11px;padding:12px 14px}
.gm-pair .gm-eyebrow{margin-bottom:.3rem}
.gm-pair strong{display:block;font-size:1.18rem;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.gm-pair small{color:var(--muted);font-size:.78rem}

div[data-testid="stMetric"]{background:var(--surface);border:1px solid var(--line);
border-radius:14px;padding:15px 18px}
div[data-testid="stMetricLabel"] p{font-size:.72rem!important;letter-spacing:.1em;
text-transform:uppercase;color:var(--muted)!important;font-weight:700}
div[data-testid="stMetricValue"]{font-size:1.85rem;letter-spacing:-.02em}

button[data-baseweb="tab"]{font-weight:600;letter-spacing:.01em}
div[data-testid="stSidebar"]{border-right:1px solid var(--line);background:#0d1524}
div[data-testid="stSidebar"] h3{font-size:.95rem}
hr{border-color:var(--line)!important;margin:.9rem 0!important}
</style>"""

STYLE = CSS[:-len("</style>")] + _ROLE_RULES + "</style>"


def chip(role: str) -> str:
    """Role badge. The colour matches the node colour on the graph."""
    return f'<span class="gm-chip" data-role="{role}"><i class="gm-dot" data-role="{role}"></i>{ROLE_LABELS[role]}</span>'


def legend() -> str:
    return '<div class="gm-legend" style="gap:8px 14px">' + "".join(chip(role) for role in ROLE_LABELS) + "</div>"


def eyebrow(text: str) -> str:
    return f'<p class="gm-eyebrow">{text}</p>'


def priority_bar(row: pd.Series) -> str:
    """Show the score as the sum of its four auditable contributions."""
    segments, items = [], []
    for column, label, colour in PRIORITY_PARTS:
        value = float(row[column])
        if value > 0:
            segments.append(f'<span style="width:{value * 100:.2f}%;background:{colour}"></span>')
        items.append(f'<span><i class="gm-dot" style="background:{colour}"></i> {label} <b>{value:.3f}</b></span>')
    return f'<div class="gm-bar">{"".join(segments)}</div><div class="gm-legend">{"".join(items)}</div>'


def money(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def flows(row: pd.Series) -> str:
    return (f'<div class="gm-pair"><div>{eyebrow("Входящие")}'
            f'<strong>{money(row.in_kzt)} ₸</strong><small>от {int(row.in_deg)} плательщиков · {int(row.in_tx)} переводов</small></div>'
            f'<div>{eyebrow("Исходящие")}'
            f'<strong>{money(row.out_kzt)} ₸</strong><small>на {int(row.out_deg)} получателей · {int(row.out_tx)} переводов</small></div></div>')


def key_values(pairs: list[tuple[str, Any]]) -> str:
    body = "".join(f"<dt>{name}</dt><dd>{value}</dd>" for name, value in pairs)
    return f'<dl class="gm-kv">{body}</dl>'
