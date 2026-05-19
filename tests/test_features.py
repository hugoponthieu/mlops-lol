import pandas as pd

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
