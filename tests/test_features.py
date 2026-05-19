import pandas as pd
import json
from pathlib import Path

from mlops.features import build_match_features


def _execute_notebook_02(project_root: Path, monkeypatch) -> pd.DataFrame:
    notebook = json.loads(Path("notebooks/02_feature_engineering.ipynb").read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    monkeypatch.chdir(project_root)

    exec_globals = {"__name__": "__main__"}
    for cell in code_cells:
        exec("".join(cell["source"]), exec_globals)

    return pd.read_csv(project_root / "artifacts" / "match_features.csv", parse_dates=["date"])


def test_build_match_features_preserves_context_columns():
    df = pd.DataFrame(
        [
            {
                "season": 1,
                "date": "2024-01-01",
                "event": "Main",
                "patch": "14.1",
                "blue_team": "AAA",
                "red_team": "BBB",
                "winner": "AAA",
                "blue_team_win": 1,
            }
        ]
    )
    df["date"] = pd.to_datetime(df["date"])

    features = build_match_features(df)

    assert "elo_diff" in features.columns
    assert "event" in features.columns
    assert "patch" in features.columns
    assert "season" in features.columns


def test_build_match_features_uses_only_prior_matches():
    df = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "blue_team": "AAA",
                "red_team": "BBB",
                "winner": "AAA",
                "blue_team_win": 1,
                "event": "Main",
                "patch": "1.0",
                "season": 1,
            },
            {
                "date": "2024-01-02",
                "blue_team": "AAA",
                "red_team": "CCC",
                "winner": "CCC",
                "blue_team_win": 0,
                "event": "Main",
                "patch": "1.0",
                "season": 1,
            },
        ]
    )
    df["date"] = pd.to_datetime(df["date"])

    features = build_match_features(df)

    first = features.iloc[0]
    second = features.iloc[1]

    assert first["elo_diff"] == 0.0
    assert first["matches_played_diff"] == 0
    assert second["matches_played_diff"] == 1
    assert second["winrate_last_5_diff"] > 0


def test_build_match_features_skips_unlabeled_matches():
    df = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "blue_team": "AAA",
                "red_team": "BBB",
                "winner": pd.NA,
                "blue_team_win": pd.NA,
                "event": "Main",
                "patch": "1.0",
                "season": 1,
            },
            {
                "date": "2024-01-02",
                "blue_team": "AAA",
                "red_team": "CCC",
                "winner": "AAA",
                "blue_team_win": 1,
                "event": "Main",
                "patch": "1.0",
                "season": 1,
            },
        ]
    )
    df["date"] = pd.to_datetime(df["date"])

    features = build_match_features(df)

    assert features.iloc[1]["matches_played_diff"] == 0
    assert features.iloc[1]["winrate_last_5_diff"] == 0.0


def test_notebook_02_runtime_contract(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "season": 1,
                "date": "2024-01-01",
                "event": "Main",
                "patch": "14.1",
                "blue_team": "AAA",
                "red_team": "BBB",
                "winner": "AAA",
                "blue_team_win": 1,
            },
            {
                "season": 1,
                "date": "2024-01-02",
                "event": "Main",
                "patch": "14.1",
                "blue_team": "AAA",
                "red_team": "CCC",
                "winner": "CCC",
                "blue_team_win": 0,
            },
        ]
    ).to_csv(artifacts_dir / "clean_matches.csv", index=False)

    match_features = _execute_notebook_02(project_root, monkeypatch)

    assert len(match_features) == 2
    assert {"season", "event", "patch", "elo_diff"}.issubset(match_features.columns)


def test_notebook_02_runtime_contract_infers_outcome_from_winner(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True)

    clean_matches = pd.DataFrame(
        [
            {
                "season": 1,
                "date": "2024-01-01",
                "event": "Main",
                "patch": "14.1",
                "blue_team": "AAA",
                "red_team": "BBB",
                "winner": "AAA",
                "blue_team_win": pd.NA,
            },
            {
                "season": 1,
                "date": "2024-01-02",
                "event": "Main",
                "patch": "14.1",
                "blue_team": "AAA",
                "red_team": "CCC",
                "winner": "AAA",
                "blue_team_win": 1,
            },
        ]
    )
    expected = build_match_features(clean_matches.assign(date=pd.to_datetime(clean_matches["date"])))
    clean_matches.to_csv(artifacts_dir / "clean_matches.csv", index=False)

    match_features = _execute_notebook_02(project_root, monkeypatch)

    actual_row = match_features.loc[
        1,
        [
            "elo_diff",
            "winrate_last_5_diff",
            "matches_played_diff",
            "blue_side_team_winrate",
        ],
    ]
    expected_row = expected.loc[1, actual_row.index]

    pd.testing.assert_series_equal(actual_row, expected_row, check_names=False)
