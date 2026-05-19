from pathlib import Path
import ast
import json

import joblib
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


def _execute_notebook(notebook_relpath: str, project_root: Path, monkeypatch) -> None:
    notebook = json.loads(Path(notebook_relpath).read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code" and cell["source"]]

    notebooks_dir = project_root / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(notebooks_dir)

    exec_globals = {"__name__": "__main__"}
    for cell in code_cells:
        exec("".join(cell["source"]), exec_globals)


def test_notebook_04_runtime_contract_writes_model_and_predictions(tmp_path, monkeypatch):
    notebook_path = Path("notebooks/04_train_probability_model.ipynb").resolve()
    source = _load_notebook_source(notebook_path)
    assert "from mlops" not in source
    assert "import sys" not in source

    project_root = tmp_path / "project"
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True)

    feature_columns = [
        "elo_diff",
        "winrate_last_5_diff",
        "winrate_last_10_diff",
        "winrate_last_20_diff",
        "matches_played_diff",
        "days_since_last_match_diff",
        "head_to_head_winrate_diff",
        "blue_side_team_winrate",
        "red_side_team_winrate",
    ]

    train_rows = [
        [10.0, 0.2, 0.1, 0.05, 1, 5, 0.5, 0.55, 0.45],
        [-15.0, -0.3, -0.2, -0.1, -2, 7, 0.0, 0.40, 0.60],
        [20.0, 0.4, 0.3, 0.2, 3, 2, 0.6, 0.65, 0.50],
        [-5.0, -0.1, -0.05, 0.0, 0, 10, 0.5, 0.50, 0.55],
        [25.0, 0.5, 0.4, 0.3, 4, 1, 0.7, 0.70, 0.45],
        [-25.0, -0.5, -0.4, -0.3, -3, 9, 0.3, 0.35, 0.65],
    ]
    valid_rows = [
        [8.0, 0.15, 0.1, 0.05, 1, 4, 0.5, 0.55, 0.50],
        [-12.0, -0.25, -0.2, -0.1, -1, 6, 0.4, 0.45, 0.60],
    ]

    pd.DataFrame(train_rows, columns=feature_columns).to_csv(
        artifacts_dir / "train_features.csv", index=False
    )
    pd.DataFrame(valid_rows, columns=feature_columns).to_csv(
        artifacts_dir / "valid_features.csv", index=False
    )
    pd.DataFrame({"blue_team_win": [1, 0, 1, 0, 1, 0]}).to_csv(
        artifacts_dir / "train_labels.csv", index=False
    )
    pd.DataFrame({"blue_team_win": [1, 0]}).to_csv(
        artifacts_dir / "valid_labels.csv", index=False
    )

    _execute_notebook("notebooks/04_train_probability_model.ipynb", project_root, monkeypatch)

    model_path = artifacts_dir / "logistic_regression_model.joblib"
    predictions_path = artifacts_dir / "validation_predictions.csv"
    manifest_path = artifacts_dir / "train_manifest.json"

    assert model_path.exists()
    assert predictions_path.exists()
    assert manifest_path.exists()

    model = joblib.load(model_path)
    sample = pd.DataFrame(valid_rows, columns=feature_columns)
    probs = model.predict_proba(sample)
    assert probs.shape == (2, 2)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()

    predictions = pd.read_csv(predictions_path)
    assert list(predictions.columns) == ["blue_win_prob", "blue_team_win"]
    assert len(predictions) == 2
    assert ((predictions["blue_win_prob"] >= 0.0) & (predictions["blue_win_prob"] <= 1.0)).all()
    assert predictions["blue_team_win"].tolist() == [1, 0]

    manifest = json.loads(manifest_path.read_text())
    assert manifest["model_name"] == "logistic_regression"
    assert manifest["train_rows"] == 6
    assert manifest["valid_rows"] == 2
    assert manifest["feature_columns"] == feature_columns

    source = _load_notebook_source(Path("notebooks/04_train_probability_model.ipynb"))
    assert "from mlops" not in source
    assert "import sys" not in source
