"""Local analyst viewer. Never recomputes; calls a model only in the optional assistant tab."""
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.narrate import ROLE_COLORS, ROLE_LABELS
from src.viewer import graph_html, read_bundle, select_neighborhood

st.set_page_config(page_title="Граф денег · Анализ сети", page_icon="◈", layout="wide")
st.markdown("""<style>
.block-container {max-width:1550px;padding-top:4.5rem}
div[data-testid="stMetric"] {background:#142033;border:1px solid #25364d;border-radius:12px;padding:14px}
h1 {letter-spacing:-1px} div[data-testid="stSidebar"] {border-right:1px solid #25364d}
</style>""", unsafe_allow_html=True)

out = Path(os.environ.get("GRAPH_OUT_DIR", str(Path(__file__).parent / "out")))
try:
    nodes, edges, clusters, top, manifest = read_bundle(out)
except (ValueError, OSError, KeyError) as error:
    st.title("Граф денег")
    st.error(str(error))
    st.stop()

config = manifest["config"]
st.caption("HACKALEM AI  /  ИЮЛЬ 2026  /  ОБЪЯСНИМЫЙ АНАЛИЗ")
st.title("Граф денег")
st.write("Кого проверить первым — и какие наблюдаемые связи это объясняют.")
st.caption("Роли — гипотезы для проверки. Скоры не являются вероятностью нарушения. Выборка ограничена банком, месяцем и глубиной обхода.")

with st.sidebar:
    st.subheader("Исследовать сеть")
    search = st.text_input("Поиск по gid", placeholder="Введите идентификатор", key="gid_search")
    cluster_filter = st.selectbox("Кластер для обзора", ["Все"] + [str(c) for c in clusters.cluster_id], key="cluster_filter")
    role_filter = st.selectbox("Роль для обзора", ["Все"] + list(ROLE_LABELS), format_func=lambda x: ROLE_LABELS.get(x, x))
    st.divider()
    st.caption("Цвет — гипотеза роли; ромб — seed. Стрелка показывает направление перевода.")
    for role, label in ROLE_LABELS.items():
        st.markdown(f'<span style="color:{ROLE_COLORS[role]}">●</span> {label}', unsafe_allow_html=True)
    st.divider()
    st.caption(f"Расчёт: {manifest['total_seconds']:.2f} с · без API")
    st.caption(f"Дата прогона UTC: {manifest['created_at_utc'][:19]}")
    st.subheader("Готовые результаты")
    st.caption("Все три CSV уже сохранены в папке результатов. Скачивание создаёт дополнительную копию.")
    with st.expander("Где лежат готовые CSV", expanded=True):
        st.code(str(out.resolve()), language=None, wrap_lines=True)
        st.caption("Папка на компьютере, где запущено приложение. Откройте её в Проводнике.")
        st.markdown("- `nodes_roles.csv` — роли всех участников\n- `top_nodes.csv` — приоритеты проверки\n- `clusters.csv` — сводка кластеров")
    with st.expander("Chrome заблокировал скачивание?"):
        st.write("Если Chrome пишет «Ваша организация заблокировала файл», скачивание ограничено политикой безопасности. Приложение не может снять это ограничение; причину блокировки должен проверить администратор браузера или компьютера.")
        st.write("Результаты расчёта уже сохранены в указанной выше папке. Блокировка скачивания не означает, что расчёт завершился с ошибкой.")
    st.download_button("Скачать роли CSV", (out / "nodes_roles.csv").read_bytes(), "nodes_roles.csv", "text/csv")
    st.download_button("Скачать топ CSV", (out / "top_nodes.csv").read_bytes(), "top_nodes.csv", "text/csv")
    st.download_button("Скачать кластеры CSV", (out / "clusters.csv").read_bytes(), "clusters.csv", "text/csv")

stats = st.columns(4)
stats[0].metric("Участники", f"{len(nodes):,}".replace(",", " "))
stats[1].metric("Связи", f"{len(edges):,}".replace(",", " "))
stats[2].metric("Кластеры", str(len(clusters)))
stats[3].metric("Граница наблюдения", str(int(nodes.truncated_by_depth.sum())))

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

network_tab, ranking_tab, clusters_tab, assistant_tab = st.tabs(["Сеть и карточка", "Приоритеты", "Кластеры и качество", "Ассистент"])
with network_tab:
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Направление потоков")
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
        st.subheader("Карточка участника")
        if selected_gid is None:
            st.info("Введите gid слева или выберите участника во вкладке «Приоритеты».")
            st.markdown("**Как читать схему**")
            st.write("Толщина ребра отражает сумму. Наведите курсор на связь, чтобы увидеть сумму и количество переводов. Перетаскивайте узлы и масштабируйте схему.")
        else:
            row = nodes.loc[nodes.gid == selected_gid].iloc[0]
            st.markdown(f"### gid {selected_gid}")
            st.write(f"**{ROLE_LABELS[row.role]}** · кластер {row.cluster_id} · место {row['rank']}")
            st.metric("Приоритет проверки", f"{row.priority_score:.3f}")
            candidate = ROLE_LABELS.get(row.best_specialized_role, "нет")
            st.caption(f"Балл роли: {row.role_score:.3f}; лучший специальный кандидат: {candidate} ({row.best_specialized_score:.3f}).")
            st.write(row.evidence)
            st.warning(row.next_request)
            if row.truncated_by_depth:
                st.caption("Исходящие за границей четвёртого колена не наблюдаются.")
            if row.is_seed:
                st.caption("Входящие seed неполны; отношения входа и выхода не используются для роли.")
            st.write(f"Вход: **{row.in_kzt:,.2f} KZT** от {row.in_deg} плательщиков")
            st.write(f"Выход: **{row.out_kzt:,.2f} KZT** на {row.out_deg} получателей")
            st.write(f"Достижим от seed: **{row.n_seed_sources}**; глубина: **{row.depth}**")
    if selected_gid is not None:
        with st.expander("Почему такая роль и приоритет", expanded=True):
            role_table = pd.DataFrame([{"Роль": ROLE_LABELS[r], "Допустима": bool(row[f"eligible_{r}"]), "Скор": row[f"score_{r}"]} for r in config["roles"]["tie_order"]])
            a, b = st.columns(2)
            a.dataframe(role_table, hide_index=True, width="stretch", column_config={"Скор": st.column_config.NumberColumn(format="%.3f")})
            b.write(row.why)
            runner_up = ROLE_LABELS.get(row.role_runner_up, "нет")
            b.caption(f"Порог специальной роли: {config['roles']['minimum_score']}; второй кандидат: {runner_up}, скор {row.runner_up_score:.3f}.")
            b.json({r: config["roles"][r] for r in config["roles"]["tie_order"]}, expanded=False)
        st.subheader("Все прямые связи")
        st.dataframe(incident.rename(columns={"src": "Отправитель", "dst": "Получатель", "sum_kzt": "Сумма KZT", "n_tx": "Переводов", "depth": "Колено"}), hide_index=True, width="stretch")
        with st.expander("Все рассчитанные метрики"):
            st.dataframe(row.astype(str).rename("Значение"), width="stretch")

with ranking_tab:
    st.subheader("С кого начать проверку")
    selected_top = st.selectbox("Участник из топ-листа", top.gid.tolist(), format_func=lambda gid: f"gid {gid}", key="top_gid")

    def open_top() -> None:
        st.session_state.gid_search = str(st.session_state.top_gid)

    st.button("Открыть карточку выбранного узла", on_click=open_top)
    st.caption("Карточка откроется во вкладке «Сеть и карточка».")
    st.dataframe(top.assign(role=top.role.map(ROLE_LABELS)).rename(columns={"rank": "Место", "gid": "gid", "role": "Роль", "priority_score": "Приоритет", "why": "Обоснование"}), hide_index=True, width="stretch", column_config={"Приоритет": st.column_config.NumberColumn(format="%.3f")})

with clusters_tab:
    st.subheader("Группы связанных участников")
    st.caption("Louvain на неориентированной проекции, суммы встречных рёбер складываются. Роли и потоки рассчитаны на направленном графе.")
    st.dataframe(clusters.rename(columns={"cluster_id": "Кластер", "n_nodes": "Узлов", "n_seed": "Seed", "sum_kzt_internal": "Внутренний оборот KZT", "top_gids": "Ключевые gid", "hypothesis": "Гипотеза"}), hide_index=True, width="stretch")
    st.subheader("Проверка исходных данных")
    report = manifest["data_report"]
    st.success(f"Агрегаты согласованы: {report['transactions']} транзакций, оборот {report['sum_kzt']:,.2f} KZT.")
    diag = manifest["diagnostics"]
    st.write(f"Компонент с изолятами: {diag['weak_components_with_isolates']}; без изолятов: {diag['weak_components_without_isolates']}. Изолированных узлов: {diag['isolated_nodes']}.")
    st.write(f"В крупнейшей компоненте {diag['largest_component']} узлов; вне неё {diag['outside_largest_component']}. Все включены в расчёт.")
    for warning in report["warnings"]:
        st.warning(warning)
    st.dataframe(nodes.role.value_counts().rename_axis("Роль").reset_index(name="Узлов"), hide_index=True)

with assistant_tab:
    st.subheader("Вопрос по сети на обычном языке")
    st.caption("Ассистент не видит граф напрямую: он вызывает функции по выгрузке и называет gid, на которых основан ответ. Роли и числа не пересчитываются.")
    # An older run may predate the llm section; the viewer must still open it.
    key_name = config.get("llm", {}).get("api_key_env", "OPENAI_API_KEY")
    if not os.environ.get(key_name):
        st.info(f"Вкладка включается переменной {key_name}. Без неё остальной инструмент работает полностью — расчёт и выгрузки не зависят от внешних сервисов.")
    else:
        question = st.text_input("Вопрос", placeholder="Кто собирает деньги с этих пятерых: 100000008686313100, ...", key="assistant_question")
        if st.button("Спросить", key="assistant_ask") and question.strip():
            from src.ai.client import LLMUnavailable
            from src.ai.tools import GraphTools, answer
            try:
                with st.spinner("Ассистент обращается к выгрузке"):
                    result = answer(question, GraphTools(nodes, edges), config)
            except LLMUnavailable as error:
                st.error(f"Ассистент недоступен: {error}")
            else:
                st.write(result["answer"])
                if result["gids"]:
                    st.caption("Узлы в ответе: " + ", ".join(str(g) for g in result["gids"]))
                with st.expander("Какие функции были вызваны"):
                    st.json(result["trace"])
