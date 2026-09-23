from pathlib import Path
import shutil

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_existing_results_remain_accessible_with_custom_folder_and_invalid_search(monkeypatch, tmp_path):
    output_dir = tmp_path / "analysis results"
    shutil.copytree(ROOT / "out", output_dir)
    monkeypatch.setenv("GRAPH_OUT_DIR", str(output_dir))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

    for invalid_search in [False, True]:
        if invalid_search:
            app.text_input(key="gid_search").set_value("not-a-number").run()
            assert any("целым" in element.value for element in app.warning)

        assert not app.exception
        assert any(element.value == "Готовые результаты" for element in app.sidebar.subheader)
        locations = [element for element in app.sidebar.expander if element.label == "Где лежат готовые CSV"]
        assert len(locations) == 1
        assert [element.value for element in locations[0].code] == [str(output_dir.resolve())]
        file_list = "\n".join(element.value for element in locations[0].markdown)
        for name in ["nodes_roles.csv", "top_nodes.csv", "clusters.csv"]:
            assert name in file_list
            assert (output_dir / name).is_file()
        assert any(element.label == "Chrome заблокировал скачивание?" for element in app.sidebar.expander)
