from pathlib import Path
import ast
import json

import pandas as pd

EXPECTED_COLUMNS = [
    "season",
    "date",
    "event",
    "patch",
    "blue_team",
    "red_team",
    "winner",
    "blue_team_win",
]


def _load_notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "".join(
        "\n".join(cell["source"])
        + "\n"
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def _execute_notebook_01(project_root: Path, monkeypatch) -> pd.DataFrame:
    notebook = json.loads(Path("notebooks/01_data_pull_and_audit.ipynb").read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code" and cell["source"]]

    monkeypatch.chdir(project_root)

    exec_globals = {"__name__": "__main__"}
    for cell in code_cells:
        exec("".join(cell["source"]), exec_globals)

    return pd.read_csv(project_root / "artifacts" / "clean_matches.csv", parse_dates=["date"])


def _extract_returned_columns(function_node: ast.FunctionDef) -> list[str]:
    for node in ast.walk(function_node):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Subscript):
            slice_node = node.value.slice
            if isinstance(slice_node, ast.List):
                columns = []
                for element in slice_node.elts:
                    if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                        raise AssertionError("artifact column selection must be a string list")
                    columns.append(element.value)
                return columns
    raise AssertionError("could not find artifact column selection in notebook helper")


def test_clean_matches_artifact_has_required_columns():
    notebook_path = Path("notebooks/01_data_pull_and_audit.ipynb")
    source = _load_notebook_source(notebook_path)
    tree = ast.parse(source)
    helper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_matches_csv"
    )

    assert "from mlops.data import load_matches" not in source
    assert "import sys" not in source
    assert 'df.to_csv(ARTIFACTS_DIR / "clean_matches.csv", index=False)' in source
    assert _extract_returned_columns(helper) == EXPECTED_COLUMNS
    assert all(f'"{column}"' in source for column in EXPECTED_COLUMNS)


def test_notebook_01_runtime_contract_from_repo_root_cwd(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "season": 1,
                "date": "2024-01-02",
                "event": " Main ",
                "patch": "14.1",
                "blue_team": " AAA ",
                "red_team": "BBB",
                "winner": " AAA ",
                "venue": "Studio",
            },
            {
                "season": 1,
                "date": "2024-01-01",
                "event": "Playoffs",
                "patch": "",
                "blue_team": "CCC",
                "red_team": "DDD",
                "winner": "DDD",
                "venue": "Arena",
            },
        ]
    ).to_csv(data_dir / "matchs_stats.csv", index=False)

    clean_matches = _execute_notebook_01(project_root, monkeypatch)

    assert (project_root / "artifacts" / "clean_matches.csv").exists()
    assert list(clean_matches.columns) == EXPECTED_COLUMNS
    assert clean_matches["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-01", "2024-01-02"]
    assert clean_matches["blue_team_win"].tolist() == [0, 1]
