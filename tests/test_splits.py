import json
from pathlib import Path

import pandas as pd

from mlops.splits import chronological_split, split_features_and_target


def test_chronological_split_preserves_time_order():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "value": [1, 2, 3, 4, 5],
        }
    )

    train_df, valid_df, test_df = chronological_split(df, train_frac=0.6, valid_frac=0.2)

    assert list(train_df["value"]) == [1, 2, 3]
    assert list(valid_df["value"]) == [4]
    assert list(test_df["value"]) == [5]
    assert train_df["date"].max() < valid_df["date"].min()
    assert valid_df["date"].max() < test_df["date"].min()


def test_chronological_split_keeps_validation_after_training():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10, freq="D")})

    train_df, valid_df, test_df = chronological_split(df, train_frac=0.6, valid_frac=0.2)

    assert train_df["date"].max() < valid_df["date"].min()
    assert valid_df["date"].max() < test_df["date"].min()


def test_chronological_split_keeps_same_day_rows_in_one_partition():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-04",
                    "2024-01-05",
                ]
            ),
            "value": [1, 2, 3, 4, 5, 6, 7],
        }
    )

    train_df, valid_df, test_df = chronological_split(df, train_frac=0.6, valid_frac=0.2)

    assert list(train_df["value"]) == [1, 2, 3, 4]
    assert list(valid_df["value"]) == [5, 6]
    assert list(test_df["value"]) == [7]
    assert set(train_df["date"]).isdisjoint(valid_df["date"])
    assert set(train_df["date"]).isdisjoint(test_df["date"])
    assert set(valid_df["date"]).isdisjoint(test_df["date"])
    assert train_df["date"].max() < valid_df["date"].min()
    assert valid_df["date"].max() < test_df["date"].min()


def test_split_features_and_target_drops_label_and_context_fields():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "blue_team": ["AAA"],
            "red_team": ["BBB"],
            "blue_team_win": [1],
            "elo_diff": [10.0],
        }
    )

    X, y = split_features_and_target(df)

    assert list(X.columns) == ["elo_diff"]
    assert list(y) == [1]


FEATURE_COLUMNS = [
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


def _load_notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "".join(
        "\n".join(cell["source"]) + "\n"
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def _execute_notebook_03(project_root: Path, monkeypatch) -> dict[str, pd.DataFrame | list[str]]:
    notebook = json.loads(Path("notebooks/03_preprocessing.ipynb").read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code" and cell["source"]]

    monkeypatch.chdir(project_root)

    exec_globals = {"__name__": "__main__"}
    for cell in code_cells:
        exec("".join(cell["source"]), exec_globals)

    outputs: dict[str, pd.DataFrame | list[str]] = {}
    for split_name in ["train", "valid", "test"]:
        outputs[f"{split_name}_features"] = pd.read_csv(project_root / "artifacts" / f"{split_name}_features.csv")
        outputs[f"{split_name}_labels"] = pd.read_csv(project_root / "artifacts" / f"{split_name}_labels.csv")
    outputs["feature_columns"] = json.loads((project_root / "artifacts" / "feature_columns.json").read_text())
    return outputs


def test_notebook_03_exports_split_artifacts_from_repo_root_cwd(tmp_path, monkeypatch):
    notebook_path = Path("notebooks/03_preprocessing.ipynb")
    source = _load_notebook_source(notebook_path)

    assert "import sys" not in source
    assert "src/mlops" not in source
    assert "from mlops" not in source
    assert 'pd.read_csv(ARTIFACTS_DIR / "match_features.csv"' in source
    assert 'ARTIFACTS_DIR / "feature_columns.json"' in source

    project_root = tmp_path / "project"
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True)

    match_features = pd.DataFrame(
        [
            {
                "season": 1,
                "date": "2024-01-02",
                "event": "Main",
                "patch": "14.1",
                "blue_team": "AAA",
                "red_team": "BBB",
                "winner": "AAA",
                "blue_team_win": 1,
                "elo_diff": 10.0,
                "winrate_last_5_diff": 0.10,
                "winrate_last_10_diff": 0.10,
                "winrate_last_20_diff": 0.10,
                "matches_played_diff": 1,
                "days_since_last_match_diff": 0,
                "head_to_head_winrate_diff": 0.10,
                "blue_side_team_winrate": 0.60,
                "red_side_team_winrate": 0.40,
            },
            {
                "season": 1,
                "date": "2024-01-01",
                "event": "Main",
                "patch": "14.1",
                "blue_team": "CCC",
                "red_team": "DDD",
                "winner": "DDD",
                "blue_team_win": 0,
                "elo_diff": -5.0,
                "winrate_last_5_diff": -0.20,
                "winrate_last_10_diff": -0.20,
                "winrate_last_20_diff": -0.20,
                "matches_played_diff": -1,
                "days_since_last_match_diff": 2,
                "head_to_head_winrate_diff": -0.10,
                "blue_side_team_winrate": 0.45,
                "red_side_team_winrate": 0.55,
            },
            {
                "season": 1,
                "date": "2024-01-05",
                "event": "Main",
                "patch": "14.1",
                "blue_team": "EEE",
                "red_team": "FFF",
                "winner": "EEE",
                "blue_team_win": 1,
                "elo_diff": 15.0,
                "winrate_last_5_diff": 0.30,
                "winrate_last_10_diff": 0.30,
                "winrate_last_20_diff": 0.30,
                "matches_played_diff": 3,
                "days_since_last_match_diff": -1,
                "head_to_head_winrate_diff": 0.20,
                "blue_side_team_winrate": 0.70,
                "red_side_team_winrate": 0.30,
            },
            {
                "season": 1,
                "date": "2024-01-02",
                "event": "Main",
                "patch": "14.1",
                "blue_team": "GGG",
                "red_team": "HHH",
                "winner": "GGG",
                "blue_team_win": 1,
                "elo_diff": 7.5,
                "winrate_last_5_diff": 0.05,
                "winrate_last_10_diff": 0.05,
                "winrate_last_20_diff": 0.05,
                "matches_played_diff": 0,
                "days_since_last_match_diff": 1,
                "head_to_head_winrate_diff": 0.00,
                "blue_side_team_winrate": 0.50,
                "red_side_team_winrate": 0.50,
            },
            {
                "season": 1,
                "date": "2024-01-03",
                "event": "Main",
                "patch": "14.1",
                "blue_team": "III",
                "red_team": "JJJ",
                "winner": "III",
                "blue_team_win": 1,
                "elo_diff": 1.0,
                "winrate_last_5_diff": 0.00,
                "winrate_last_10_diff": 0.00,
                "winrate_last_20_diff": 0.00,
                "matches_played_diff": 0,
                "days_since_last_match_diff": 0,
                "head_to_head_winrate_diff": 0.00,
                "blue_side_team_winrate": 0.50,
                "red_side_team_winrate": 0.50,
            },
            {
                "season": 1,
                "date": "2024-01-04",
                "event": "Main",
                "patch": "14.1",
                "blue_team": "KKK",
                "red_team": "LLL",
                "winner": "LLL",
                "blue_team_win": 0,
                "elo_diff": 2.0,
                "winrate_last_5_diff": 0.01,
                "winrate_last_10_diff": 0.01,
                "winrate_last_20_diff": 0.01,
                "matches_played_diff": 1,
                "days_since_last_match_diff": 1,
                "head_to_head_winrate_diff": 0.05,
                "blue_side_team_winrate": 0.51,
                "red_side_team_winrate": 0.49,
            },
            {
                "season": 1,
                "date": "2024-01-04",
                "event": "Main",
                "patch": "14.1",
                "blue_team": "OOO",
                "red_team": "PPP",
                "winner": "OOO",
                "blue_team_win": 1,
                "elo_diff": 4.0,
                "winrate_last_5_diff": 0.04,
                "winrate_last_10_diff": 0.04,
                "winrate_last_20_diff": 0.04,
                "matches_played_diff": 2,
                "days_since_last_match_diff": 3,
                "head_to_head_winrate_diff": 0.12,
                "blue_side_team_winrate": 0.58,
                "red_side_team_winrate": 0.42,
            },
            {
                "season": 1,
                "date": "2024-01-06",
                "event": "Main",
                "patch": "14.1",
                "blue_team": "MMM",
                "red_team": "NNN",
                "winner": pd.NA,
                "blue_team_win": pd.NA,
                "elo_diff": 3.0,
                "winrate_last_5_diff": 0.02,
                "winrate_last_10_diff": 0.02,
                "winrate_last_20_diff": 0.02,
                "matches_played_diff": 2,
                "days_since_last_match_diff": 2,
                "head_to_head_winrate_diff": 0.10,
                "blue_side_team_winrate": 0.52,
                "red_side_team_winrate": 0.48,
            },
        ]
    )
    match_features.to_csv(artifacts_dir / "match_features.csv", index=False)

    outputs = _execute_notebook_03(project_root, monkeypatch)

    labeled_df = match_features.assign(date=pd.to_datetime(match_features["date"]))
    labeled_df = labeled_df[labeled_df["blue_team_win"].notna()].copy()
    labeled_df["blue_team_win"] = labeled_df["blue_team_win"].astype("Int64")
    expected_train, expected_valid, expected_test = chronological_split(labeled_df)

    assert outputs["feature_columns"] == FEATURE_COLUMNS
    assert len(outputs["train_features"]) == 4
    assert len(outputs["valid_features"]) == 2
    assert len(outputs["test_features"]) == 1

    for split_name, expected_df in [
        ("train", expected_train),
        ("valid", expected_valid),
        ("test", expected_test),
    ]:
        expected_features = expected_df[FEATURE_COLUMNS].reset_index(drop=True)
        expected_labels = expected_df[["blue_team_win"]].astype("int64").reset_index(drop=True)

        pd.testing.assert_frame_equal(outputs[f"{split_name}_features"], expected_features)
        pd.testing.assert_frame_equal(outputs[f"{split_name}_labels"], expected_labels)
