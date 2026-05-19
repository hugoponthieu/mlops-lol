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


def _execute_notebook(
    notebook_relpath: str,
    project_root: Path,
    monkeypatch,
    cwd_subdir: str | None = None,
) -> None:
    notebook = json.loads(Path(notebook_relpath).read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code" and cell["source"]]

    target_cwd = project_root if cwd_subdir is None else project_root / cwd_subdir
    target_cwd.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(target_cwd)

    exec_globals = {"__name__": "__main__"}
    for cell in code_cells:
        exec("".join(cell["source"]), exec_globals)


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

    _execute_notebook("notebooks/01_data_pull_and_audit.ipynb", project_root, monkeypatch)
    clean_matches = pd.read_csv(project_root / "artifacts" / "clean_matches.csv", parse_dates=["date"])

    assert (project_root / "artifacts" / "clean_matches.csv").exists()
    assert list(clean_matches.columns) == EXPECTED_COLUMNS
    assert clean_matches["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-01", "2024-01-02"]
    assert clean_matches["blue_team_win"].tolist() == [0, 1]


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

    _execute_notebook(str(notebook_path), project_root, monkeypatch, cwd_subdir="notebooks")

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


def test_notebook_05_runtime_contract_writes_metrics_predictions_and_plot(tmp_path, monkeypatch):
    notebook_path = Path("notebooks/05_evaluation_and_calibration.ipynb").resolve()
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

    import numpy as np
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    n_train = 60
    n_test = 30
    train_X = pd.DataFrame(
        rng.normal(size=(n_train, len(feature_columns))), columns=feature_columns
    )
    train_y = (train_X["elo_diff"] > 0).astype(int)
    test_X = pd.DataFrame(
        rng.normal(size=(n_test, len(feature_columns))), columns=feature_columns
    )
    test_y = (test_X["elo_diff"] > 0).astype(int)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_include=["number"])),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(train_X, train_y)
    joblib.dump(model, artifacts_dir / "logistic_regression_model.joblib")

    test_X.to_csv(artifacts_dir / "test_features.csv", index=False)
    pd.DataFrame({"blue_team_win": test_y.tolist()}).to_csv(
        artifacts_dir / "test_labels.csv", index=False
    )

    _execute_notebook(str(notebook_path), project_root, monkeypatch, cwd_subdir="notebooks")

    metrics_path = artifacts_dir / "evaluation_metrics.json"
    predictions_path = artifacts_dir / "test_predictions.csv"
    plot_path = artifacts_dir / "plots" / "calibration_curve.png"

    assert metrics_path.exists()
    assert predictions_path.exists()
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0

    metrics = json.loads(metrics_path.read_text())
    assert set(metrics.keys()) == {"log_loss", "brier_score", "roc_auc", "accuracy_50"}
    for key, value in metrics.items():
        assert isinstance(value, float), f"metric {key!r} must be a float"

    predictions = pd.read_csv(predictions_path)
    assert list(predictions.columns) == ["blue_win_prob", "blue_team_win"]
    assert len(predictions) == n_test
    assert ((predictions["blue_win_prob"] >= 0.0) & (predictions["blue_win_prob"] <= 1.0)).all()
    assert set(predictions["blue_team_win"].unique()).issubset({0, 1})


def test_notebook_06_runtime_contract_exports_prediction_artifact(tmp_path, monkeypatch):
    notebook_path = Path("notebooks/06_predict_match.ipynb").resolve()
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
    (artifacts_dir / "feature_columns.json").write_text(json.dumps(feature_columns))

    history_rows = [
        {"season": 1, "date": "2024-01-01", "event": "Main", "patch": "1.0",
         "blue_team": "AAA", "red_team": "BBB", "winner": "AAA", "blue_team_win": 1},
        {"season": 1, "date": "2024-01-02", "event": "Main", "patch": "1.0",
         "blue_team": "AAA", "red_team": "CCC", "winner": "AAA", "blue_team_win": 1},
        {"season": 1, "date": "2024-01-03", "event": "Main", "patch": "1.0",
         "blue_team": "DDD", "red_team": "AAA", "winner": "AAA", "blue_team_win": 0},
        {"season": 1, "date": "2024-01-04", "event": "Main", "patch": "1.0",
         "blue_team": "BBB", "red_team": "CCC", "winner": "BBB", "blue_team_win": 1},
        {"season": 1, "date": "2024-01-05", "event": "Main", "patch": "1.0",
         "blue_team": "CCC", "red_team": "DDD", "winner": "DDD", "blue_team_win": 0},
        {"season": 1, "date": "2024-01-06", "event": "Main", "patch": "1.0",
         "blue_team": "AAA", "red_team": "DDD", "winner": "AAA", "blue_team_win": 1},
    ]
    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(artifacts_dir / "clean_matches.csv", index=False)

    import numpy as np
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    n_train = 60
    train_X = pd.DataFrame(
        rng.normal(size=(n_train, len(feature_columns))), columns=feature_columns
    )
    train_y = (train_X["elo_diff"] > 0).astype(int)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_include=["number"])),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(train_X, train_y)
    joblib.dump(model, artifacts_dir / "logistic_regression_model.joblib")

    _execute_notebook(str(notebook_path), project_root, monkeypatch, cwd_subdir="notebooks")

    result_path = artifacts_dir / "predicted_match_result.csv"
    assert result_path.exists()

    result = pd.read_csv(result_path)
    assert list(result.columns) == [
        "date",
        "blue_team",
        "red_team",
        "blue_win_prob",
        "red_win_prob",
        "predicted_winner",
    ]
    assert len(result) == 1

    row = result.iloc[0]
    assert 0.0 <= row["blue_win_prob"] <= 1.0
    assert 0.0 <= row["red_win_prob"] <= 1.0
    assert abs((row["blue_win_prob"] + row["red_win_prob"]) - 1.0) < 1e-9
    assert row["predicted_winner"] in {row["blue_team"], row["red_team"]}
