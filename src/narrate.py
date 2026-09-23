"""Offline, fact-based hypotheses. No text implies guilt."""
from typing import Any
import pandas as pd

ROLE_LABELS = {"consolidator": "Консолидация", "distributor": "Распределение", "transit": "Транзит",
               "terminal": "Конечный получатель", "coordinator": "Координация", "peripheral": "Периферия"}
ROLE_COLORS = {"consolidator": "#f59e0b", "distributor": "#3b82f6", "transit": "#14b8a6",
               "terminal": "#a78bfa", "coordinator": "#ef6461", "peripheral": "#94a3b8"}


def add_evidence(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    df = frame.copy()
    messages = []
    for r in df.itertuples(index=False):
        if r.role == "consolidator":
            text = f"Признаки консолидации: {r.in_deg} плательщиков; далее {r.pass_through:.0%} наблюдаемого входа. Требует проверки."
        elif r.role == "distributor":
            text = f"Признаки распределения: {r.out_deg} получателей, {r.out_kzt:,.0f} KZT исходящих. Требует проверки."
        elif r.role == "transit":
            text = f"Признаки транзита: вход {r.in_kzt:,.0f} KZT; выход/вход {r.pass_through:.2f}. Это месячная выборка, требует проверки."
        elif r.role == "terminal":
            text = f"Гипотеза конечного получателя: вход {r.in_kzt:,.0f} KZT, исходящих 0; глубина {r.depth}. Полный баланс неизвестен."
        elif r.role == "coordinator":
            text = f"Гипотеза координации: посредничество {r.betweenness:.5f}, достижим от {r.n_seed_sources} seed. Требует проверки."
        else:
            text = f"Недостаточно признаков специальной роли: входящих связей {r.in_deg}, исходящих {r.out_deg}. Не вывод о безвредности."
        if r.truncated_by_depth:
            text = f"Входящих связей {r.in_deg}; обход обрезан на глубине {r.depth}. Исходящие неизвестны: нужен дозапрос. Специальная роль не подтверждена."
        elif r.is_seed:
            text += " Входящие seed неполны."
        if len(text) > config["export"]["evidence_max_chars"]:
            raise ValueError(f"Обоснование gid={r.gid} длиннее ограничения ТЗ")
        messages.append(text)
    df["evidence"] = messages
    return df
