from pathlib import Path
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_app_search_and_filters():
    nodes = pd.read_parquet(ROOT / "out/node_features.parquet")
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    for mask in [nodes.is_isolated, nodes.truncated_by_depth, nodes.role.eq("coordinator")]:
        gid = int(nodes.loc[mask, "gid"].iloc[0])
        app.text_input(key="gid_search").set_value(str(gid)).run()
        assert not app.exception
        assert any(f"gid {gid}" in element.value for element in app.markdown)
    app.text_input(key="gid_search").set_value("not-a-number").run()
    assert not app.exception and any("целым" in element.value for element in app.warning)
    app.text_input(key="gid_search").set_value(str(int(nodes.gid.max()) + 1)).run()
    assert not app.exception and any("отсутствует" in element.value for element in app.warning)
    app.text_input(key="gid_search").set_value("").run()
    app.selectbox(key="cluster_filter").select(str(int(nodes.cluster_id.max()))).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception and app.text_input(key="gid_search").value


def test_app_missing_results(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPH_OUT_DIR", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception and any("Результаты не найдены" in element.value for element in app.error)
