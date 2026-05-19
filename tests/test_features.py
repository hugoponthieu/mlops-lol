import pandas as pd
import json
from pathlib import Path

from mlops.features import build_match_features


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


def test_notebook_02_runtime_contract():
    notebook = json.loads(Path("notebooks/02_feature_engineering.ipynb").read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    helper_source = "".join(code_cells[0]["source"])
    orchestration_sources = ["".join(cell["source"]) for cell in code_cells[1:]]
    all_source = "\n".join("".join(cell["source"]) for cell in code_cells)

    assert "sys.path" not in all_source
    assert "from mlops.features import" not in all_source
    assert "import mlops.features" not in all_source
    assert 'pd.read_csv(ARTIFACTS_DIR / "clean_matches.csv"' in all_source
    assert '"season": row.season' in helper_source
    assert '"event": row.event' in helper_source
    assert '"patch": row.patch' in helper_source
    assert any("match_features = build_match_features_local(clean_matches)" in source for source in orchestration_sources)
    assert any(
        'match_features.to_csv(ARTIFACTS_DIR / "match_features.csv", index=False)' in source
        for source in orchestration_sources
    )
