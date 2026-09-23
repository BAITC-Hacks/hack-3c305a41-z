"""Local analyst viewer. Never recomputes; calls a model only in the optional assistant tab."""
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.narrate import ROLE_COLORS, ROLE_LABELS
from src.ui import STYLE, chip, eyebrow, flows, key_values, legend, money, priority_bar
from src.viewer import graph_html, read_bundle, select_neighborhood

st.set_page_config(page_title="Граф денег · Анализ сети", page_icon="◈", layout="wide")
st.markdown(STYLE, unsafe_allow_html=True)

out = Path(os.environ.get("GRAPH_OUT_DIR", str(Path(__file__).parent / "out")))
try:
    nodes, edges, clusters, top, resilience, manifest = read_bundle(out)
except (ValueError, OSError, KeyError) as error:
    st.title("Граф денег")
    st.error(str(error))
    st.stop()

config = manifest["config"]
header, badge = st.columns([3, 1])
with header:
    st.markdown(eyebrow("HackAlem AI · июль 2026 · объяснимый анализ"), unsafe_allow_html=True)
    st.title("Граф денег")
    st.markdown('<p class="gm-lede">Кого из участников сети проверить первым — и какие наблюдаемые '
                'числа это объясняют. Роли присваиваются правилами с порогами, не моделью.</p>', unsafe_allow_html=True)
with badge:
    st.markdown(
        f'<div class="gm-card" style="margin-top:34px">{eyebrow("Последний расчёт")}'
        f'{key_values([("Длительность", f"{manifest["total_seconds"]:.2f} с"), ("Лимит ТЗ", "300 с"),
                       ("Обращений к API", "нет"), ("Дата UTC", manifest["created_at_utc"][:10])])}</div>',
        unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Исследовать сеть")
    search = st.text_input("Поиск по gid", placeholder="Введите идентификатор", key="gid_search")
    cluster_filter = st.selectbox("Кластер для обзора", ["Все"] + [str(c) for c in clusters.cluster_id], key="cluster_filter")
    role_filter = st.selectbox("Роль для обзора", ["Все"] + list(ROLE_LABELS), format_func=lambda x: ROLE_LABELS.get(x, x))
    st.divider()
    st.markdown(eyebrow("Гипотезы ролей") + legend(), unsafe_allow_html=True)
    st.caption("Цвет узла на схеме совпадает с цветом роли. Ромб — участник из исходного списка (seed). Стрелка показывает направление перевода.")
    st.divider()
    st.subheader("Готовые результаты")
    st.caption("Все три CSV уже сохранены в папке результатов. Скачивание создаёт дополнительную копию.")
    with st.expander("Где лежат готовые CSV", expanded=False):
        st.code(str(out.resolve()), language=None, wrap_lines=True)
        st.caption("Папка на компьютере, где запущено приложение. Откройте её в Проводнике.")
        st.markdown("- `nodes_roles.csv` — роли всех участников\n- `top_nodes.csv` — приоритеты проверки\n- `clusters.csv` — сводка кластеров")
    with st.expander("Chrome заблокировал скачивание?"):
        st.write("Если Chrome пишет «Ваша организация заблокировала файл», скачивание ограничено политикой безопасности. Приложение не может снять это ограничение; причину блокировки должен проверить администратор браузера или компьютера.")
        st.write("Результаты расчёта уже сохранены в указанной выше папке. Блокировка скачивания не означает, что расчёт завершился с ошибкой.")
    st.download_button("Скачать роли CSV", (out / "nodes_roles.csv").read_bytes(), "nodes_roles.csv", "text/csv", width="stretch")
    st.download_button("Скачать топ CSV", (out / "top_nodes.csv").read_bytes(), "top_nodes.csv", "text/csv", width="stretch")
    st.download_button("Скачать кластеры CSV", (out / "clusters.csv").read_bytes(), "clusters.csv", "text/csv", width="stretch")

focus = int((nodes.role.isin(["consolidator", "coordinator"])).sum())
stats = st.columns(5)
stats[0].metric("Участники", money(len(nodes)), help="Все узлы наблюдаемой сети, включая изолированные. Исходный список — 81 клиент.")
stats[1].metric("Связи", money(len(edges)), help="Пары плательщик → получатель, агрегированные за июль 2026.")
stats[2].metric("Точки сбора", money(focus), help="Узлы с ролью «Консолидация» или «Координация» — те, ради кого строился анализ.")
stats[3].metric("Группы", money(len(clusters)), help="Сообщества Louvain на неориентированной проекции графа.")
stats[4].metric("Граница наблюдения", money(int(nodes.truncated_by_depth.sum())),
                help="Узлы на четвёртом колене без исходящих. Это край выгрузки, а не доказательство, что деньги осели.")

selected_gid = None
if search.strip():
    try:
        selected_gid = int(search.strip())
    except ValueError:
        st.warning("gid должен быть целым числом.")
        st.stop()
    if selected_gid not in set(nodes.gid):
        st.warning(f"gid {selected_gid} отсутствует в этой выгрузке.")
        st.stop()

filtered = nodes
if cluster_filter != "Все":
    filtered = filtered[filtered.cluster_id == int(cluster_filter)]
if role_filter != "Все":
    filtered = filtered[filtered.role == role_filter]

network_tab, ranking_tab, clusters_tab, resilience_tab, assistant_tab = st.tabs(
    ["Сеть и карточка", "Приоритеты", "Кластеры и качество", "Устойчивость сети", "Ассистент"])

with network_tab:
    left, right = st.columns([2.05, 1], gap="medium")
    with left:
        st.markdown(eyebrow("Направление потоков"), unsafe_allow_html=True)
        if selected_gid is not None:
            shown, links, incident = select_neighborhood(nodes, edges, selected_gid, config["viewer"]["max_nodes"])
            total_neighbors = len(set(incident.src) | set(incident.dst) | {selected_gid})
            st.caption(f"gid {selected_gid}: показано {len(shown)} из {total_neighbors} узлов ближайшего окружения. Поиск охватывает всю выгрузку независимо от фильтров обзора.")
            if len(shown) < total_neighbors:
                st.info("Граф сокращён по приоритету. Все прямые связи доступны в таблице ниже.")
        else:
            shown = filtered.sort_values(["priority_score", "gid"], ascending=[False, True]).head(config["viewer"]["max_nodes"])
            links = edges[edges.src.isin(shown.gid) & edges.dst.isin(shown.gid)]
            st.caption(f"Обзор: {len(shown)} из {len(filtered)} узлов по приоритету; показаны связи между ними. Для полного окружения найдите gid.")
        if shown.empty:
            st.info("По выбранным фильтрам нет узлов.")
        else:
            components.html(graph_html(shown, links, config, selected_gid), height=config["viewer"]["height"] + 10, scrolling=False)

    with right:
        st.markdown(eyebrow("Карточка участника"), unsafe_allow_html=True)
        if selected_gid is None:
            st.markdown(
                f'<div class="gm-card">{eyebrow("Как пользоваться")}'
                '<p class="gm-lede">Введите gid в поле слева или выберите участника во вкладке «Приоритеты» — '
                'здесь появится роль, разложение приоритета и все связи.</p>'
                '<p class="gm-lede" style="margin-top:10px">На схеме размер узла отражает приоритет, толщина стрелки — сумму. '
                'Наведите курсор на связь, чтобы увидеть сумму и число переводов.</p></div>', unsafe_allow_html=True)
        else:
            row = nodes.loc[nodes.gid == selected_gid].iloc[0]
            limitation = ""
            if row.truncated_by_depth:
                limitation = ('<div class="gm-note"><b>Край выгрузки.</b> Обход завершён на четвёртом колене: '
                              'исходящие переводы могут существовать вне данных. Нужен дозапрос.</div>')
            elif row.is_seed:
                limitation = ('<div class="gm-note"><b>Участник исходного списка.</b> Входящие переводы неполны, '
                              'поэтому отношение выхода к входу для роли не используется.</div>')
            st.markdown(
                f'<div class="gm-card"><div class="gm-id">gid {selected_gid}</div>'
                f'{chip(row.role)} <span class="gm-rank">место {int(row["rank"])} из {len(nodes)}</span> '
                f'<span class="gm-rank">группа {int(row.cluster_id)}</span>'
                f'<div style="margin-top:16px">{eyebrow(f"Приоритет проверки · {row.priority_score:.3f}")}'
                f'{priority_bar(row)}</div>'
                f'<div style="margin-top:14px">{eyebrow("Обоснование роли")}'
                f'<div class="gm-quote">{row.evidence}</div></div>'
                f'{limitation}'
                f'<div style="margin-top:14px">{flows(row)}</div>'
                f'<div style="margin-top:12px">{key_values([
                    ("Уверенность в роли", f"{row.role_score:.3f}"),
                    ("Второй кандидат", f"{ROLE_LABELS.get(row.role_runner_up, "нет")} · {row.runner_up_score:.3f}"),
                    ("Достижим от seed", int(row.n_seed_sources)),
                    ("Колено обхода", int(row.depth))])}</div></div>',
                unsafe_allow_html=True)
            timing = []
            if pd.notna(row.hold_days):
                timing.append(("Медиана удержания", f"{row.hold_days:.0f} дн."))
            if pd.notna(row.fast_pass_share):
                timing.append(("Ушло в течение 2 дней", f"{row.fast_pass_share:.0%} исходящих"))
            if int(row.same_day_inflows) > 1:
                timing.append(("Плательщиков в один день", int(row.same_day_inflows)))
            if int(row.active_days) > 0:
                timing.append(("Дней с операциями", int(row.active_days)))
            if timing:
                marks = []
                if bool(row.fast_pass):
                    marks.append("сквозной транзит")
                if bool(row.synchronised_inflow):
                    marks.append("синхронные поступления")
                tail = (f'<p class="gm-lede" style="margin-top:8px">Отмечено: {", ".join(marks)}. '
                        'Временные признаки сообщаются рядом с ролью и не влияют на её присвоение.</p>') if marks else ""
                st.markdown(f'<div class="gm-card">{eyebrow("Временной профиль")}{key_values(timing)}{tail}</div>',
                            unsafe_allow_html=True)
            st.caption(f"Что запросить дальше: {row.next_request}")

    if selected_gid is not None:
        with st.expander("Почему именно эта роль — правила и пороги", expanded=False):
            rules, explanation = st.columns([1, 1], gap="medium")
            role_table = pd.DataFrame([{"Роль": ROLE_LABELS[r], "Правило выполнено": bool(row[f"eligible_{r}"]),
                                        "Скор": row[f"score_{r}"]} for r in config["roles"]["tie_order"]])
            rules.dataframe(role_table.sort_values("Скор", ascending=False), hide_index=True, width="stretch",
                            column_config={"Скор": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1)})
            rules.caption(f"Роль получает наибольший скор среди правил, условия которых выполнены. "
                          f"Порог специальной роли — {config['roles']['minimum_score']}.")
            explanation.markdown(f'{eyebrow("Разложение приоритета")}<div class="gm-quote">{row.why}</div>', unsafe_allow_html=True)
            explanation.caption("Пороги ниже взяты из config.yaml — их можно изменить без правки кода.")
            explanation.json({r: config["roles"][r] for r in config["roles"]["tie_order"]}, expanded=False)

        st.markdown(eyebrow("Все прямые связи участника"), unsafe_allow_html=True)
        st.dataframe(incident.rename(columns={"src": "Отправитель", "dst": "Получатель", "sum_kzt": "Сумма KZT",
                                              "n_tx": "Переводов", "depth": "Колено"}),
                     hide_index=True, width="stretch",
                     column_config={"Сумма KZT": st.column_config.NumberColumn(format="%.2f")})
        with st.expander("Все рассчитанные метрики участника"):
            st.dataframe(row.astype(str).rename("Значение"), width="stretch")

with ranking_tab:
    st.markdown(eyebrow("С кого начать проверку"), unsafe_allow_html=True)
    st.markdown('<p class="gm-lede">Список отсортирован по приоритету. Приоритет складывается из четырёх '
                'слагаемых: вес роли, связь с исходным списком, объём и посредничество.</p>', unsafe_allow_html=True)

    leaders = "".join(
        f'<div style="flex:1 1 230px" class="gm-card">{eyebrow(f"Место {int(r["rank"])}")}'
        f'<div class="gm-id" style="font-size:1.15rem">gid {int(r.gid)}</div>{chip(r.role)}'
        f'<div style="margin-top:10px">{priority_bar(r)}</div></div>'
        for _, r in nodes.nsmallest(3, "rank").iterrows())
    st.markdown(f'<div class="gm-pair" style="gap:14px">{leaders}</div>', unsafe_allow_html=True)

    selected_top = st.selectbox("Участник из топ-листа", top.gid.tolist(), format_func=lambda gid: f"gid {gid}", key="top_gid")

    def open_top() -> None:
        st.session_state.gid_search = str(st.session_state.top_gid)

    st.button("Открыть карточку выбранного узла", on_click=open_top, type="primary", key="open_top")
    st.caption("Карточка откроется во вкладке «Сеть и карточка».")
    st.dataframe(top.assign(role=top.role.map(ROLE_LABELS)).rename(
        columns={"rank": "Место", "gid": "gid", "role": "Роль", "priority_score": "Приоритет", "why": "Обоснование"}),
        hide_index=True, width="stretch",
        column_config={"Приоритет": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1)})

with clusters_tab:
    st.markdown(eyebrow("Группы связанных участников"), unsafe_allow_html=True)
    st.markdown('<p class="gm-lede">Louvain на неориентированной проекции: суммы встречных рёбер складываются. '
                'Роли и направление потоков при этом рассчитаны на направленном графе.</p>', unsafe_allow_html=True)
    st.dataframe(clusters.rename(columns={"cluster_id": "Группа", "n_nodes": "Участников", "n_seed": "Из них seed",
                                          "sum_kzt_internal": "Внутренний оборот KZT", "top_gids": "Ключевые gid",
                                          "hypothesis": "Гипотеза назначения"}),
                 hide_index=True, width="stretch",
                 column_config={"Внутренний оборот KZT": st.column_config.NumberColumn(format="%.2f"),
                                "Участников": st.column_config.ProgressColumn(format="%d", min_value=0,
                                                                              max_value=int(clusters.n_nodes.max()))})

    st.markdown(eyebrow("Распределение ролей"), unsafe_allow_html=True)
    counts = nodes.role.value_counts()
    bars = "".join(
        f'<div style="margin-bottom:9px">{chip(role)}'
        f'<span style="float:right;font-variant-numeric:tabular-nums;font-weight:600">{money(int(counts.get(role, 0)))}</span>'
        f'<div class="gm-bar" style="margin:6px 0 0"><span style="width:{counts.get(role, 0) / len(nodes) * 100:.1f}%;'
        f'background:{ROLE_COLORS[role]}"></span></div></div>'
        for role in ROLE_LABELS)
    st.markdown(f'<div class="gm-card">{bars}</div>', unsafe_allow_html=True)

    st.markdown(eyebrow("Проверка исходных данных"), unsafe_allow_html=True)
    report, diag = manifest["data_report"], manifest["diagnostics"]
    st.success(f"Агрегаты согласованы: {report['transactions']} транзакций, оборот {report['sum_kzt']:,.2f} KZT.")
    st.markdown(
        f'<div class="gm-card">{key_values([
            ("Слабосвязных компонент", diag["weak_components_without_isolates"]),
            ("Изолированных участников", diag["isolated_nodes"]),
            ("В крупнейшей компоненте", money(diag["largest_component"])),
            ("Вне крупнейшей компоненты", money(diag["outside_largest_component"]))])}'
        '<p class="gm-lede" style="margin-top:10px">Все перечисленные участники включены в расчёт '
        'и получили роль — сеть не монолитна, и это учтено.</p></div>', unsafe_allow_html=True)
    for warning in report["warnings"]:
        st.warning(warning)

with resilience_tab:
    st.markdown(eyebrow("Что будет с сетью, если изъять верх списка"), unsafe_allow_html=True)
    st.markdown('<p class="gm-lede">Проверка смысла самого рейтинга. Узлы изымаются двумя способами: по нашему '
                'приоритету и случайно, случайный вариант усреднён по нескольким розыгрышам с фиксированными сидами. '
                'Если бы наш порядок был неинформативным, обе кривые совпали бы.</p>', unsafe_allow_html=True)

    summary = manifest.get("resilience", {})
    step = int(summary.get("headline_step", 20))
    if summary.get("priority_largest_component"):
        head = st.columns(3)
        head[0].metric(f"Крупнейший фрагмент после изъятия {step}", money(summary["priority_largest_component"]),
                       delta=f"{summary['priority_largest_component'] - summary['random_largest_component']:.0f} против случайного",
                       delta_color="inverse",
                       help="Чем меньше остался крупнейший связный фрагмент, тем сильнее разрушена сеть.")
        head[1].metric("Фрагментов сети", f"{summary['priority_fragments']:.0f}",
                       delta=f"против {summary['random_fragments']:.1f} при случайном изъятии",
                       help="Связные группы от двух участников. Рост означает распад сети на изолированные куски.")
        head[2].metric("Оборот в крупнейшем фрагменте", f"{summary['priority_turnover_share']:.0%}",
                       delta=f"против {summary['random_turnover_share']:.0%} при случайном",
                       delta_color="inverse",
                       help="Доля наблюдаемого оборота, оставшаяся внутри уцелевшего ядра сети.")
        st.markdown(
            f'<div class="gm-card">{eyebrow("Как это читать")}'
            f'<p class="gm-lede">Изъятие {step} участников из верха списка отрезает от ядра сети на '
            f'<b>{summary["advantage_nodes_detached"]}</b> участников больше, чем изъятие такого же числа случайных, '
            f'и оставляет в ядре {summary["priority_turnover_share"]:.0%} оборота вместо '
            f'{summary["random_turnover_share"]:.0%}. Это измеримое подтверждение того, что список указывает '
            'на структуру, а не просто на крупные суммы.</p>'
            '<p class="gm-lede" style="margin-top:10px">Оценка описывает наблюдаемую сеть переводов за июль 2026 '
            'и не является прогнозом последствий каких-либо действий в отношении клиентов.</p></div>',
            unsafe_allow_html=True)

    chart = resilience.assign(Стратегия=resilience.strategy.map({"priority": "По приоритету", "random": "Случайно"}))
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(eyebrow("Крупнейший связный фрагмент"), unsafe_allow_html=True)
        st.line_chart(chart, x="removed", y="largest_component", color="Стратегия", height=300)
    with right:
        st.markdown(eyebrow("Число фрагментов сети"), unsafe_allow_html=True)
        st.line_chart(chart, x="removed", y="components_min2", color="Стратегия", height=300)

    st.markdown(eyebrow("Полная таблица"), unsafe_allow_html=True)
    st.dataframe(chart.rename(columns={"removed": "Изъято узлов", "largest_component": "Крупнейший фрагмент",
                                       "components_min2": "Фрагментов", "seeds_outside_largest": "Seed вне ядра",
                                       "turnover_share_in_largest": "Доля оборота в ядре"})
                 .drop(columns=["strategy"]),
                 hide_index=True, width="stretch",
                 column_config={"Доля оборота в ядре": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})
    st.caption("Источник: out/resilience.csv, пересчитывается вместе с остальными выгрузками.")

with assistant_tab:
    st.markdown(eyebrow("Вопрос по сети на обычном языке"), unsafe_allow_html=True)
    st.markdown('<p class="gm-lede">Ассистент не видит граф напрямую: он вызывает функции по готовой выгрузке '
                'и называет gid, на которых основан ответ. Роли и числа при этом не пересчитываются.</p>', unsafe_allow_html=True)
    # An older run may predate the llm section; the viewer must still open it.
    key_name = config.get("llm", {}).get("api_key_env", "OPENAI_API_KEY")
    if not os.environ.get(key_name):
        st.info(f"Вкладка включается переменной {key_name}. Без неё остальной инструмент работает полностью — "
                "расчёт, выгрузки и схема сети не зависят от внешних сервисов.")
        st.markdown(
            f'<div class="gm-card">{eyebrow("Что можно будет спросить")}'
            '<p class="gm-lede">· Кто собирает деньги с этих пятерых: 100000008686313100, ...<br>'
            '· Покажи путь денег от gid A до gid B<br>'
            '· Какие точки консолидации в группе 4 и почему они там<br>'
            '· С кем связан gid X и сколько он получил</p></div>', unsafe_allow_html=True)
    else:
        question = st.text_input("Вопрос", placeholder="Кто собирает деньги с этих пятерых: 100000008686313100, ...", key="assistant_question")
        if st.button("Спросить", key="assistant_ask", type="primary") and question.strip():
            from src.ai.client import LLMUnavailable
            from src.ai.tools import GraphTools, answer
            try:
                with st.spinner("Ассистент обращается к выгрузке"):
                    result = answer(question, GraphTools(nodes, edges), config)
            except LLMUnavailable as error:
                st.error(f"Ассистент недоступен: {error}")
            else:
                st.markdown(f'<div class="gm-card"><div class="gm-quote">{result["answer"]}</div></div>', unsafe_allow_html=True)
                if result["gids"]:
                    st.caption("Узлы в ответе: " + ", ".join(str(g) for g in result["gids"]))
                with st.expander("Какие функции были вызваны"):
                    st.json(result["trace"])
