import pandas as pd

from mlops.predict import build_upcoming_match_features, predict_match_result


def _history_df():
    df = pd.DataFrame(
        [
            {
                "season": 1,
                "date": "2024-01-01",
                "event": "Main",
                "patch": "1.0",
                "blue_team": "AAA",
                "red_team": "BBB",
                "winner": "AAA",
                "blue_team_win": 1,
            },
            {
                "season": 1,
                "date": "2024-01-02",
                "event": "Main",
                "patch": "1.0",
                "blue_team": "AAA",
                "red_team": "CCC",
                "winner": "AAA",
                "blue_team_win": 1,
            },
            {
                "season": 1,
                "date": "2024-01-03",
                "event": "Main",
                "patch": "1.0",
                "blue_team": "DDD",
                "red_team": "AAA",
                "winner": "AAA",
                "blue_team_win": 0,
            },
            {
                "season": 1,
                "date": "2024-01-04",
                "event": "Main",
                "patch": "1.0",
                "blue_team": "BBB",
                "red_team": "CCC",
                "winner": "BBB",
                "blue_team_win": 1,
            },
        ]
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_build_upcoming_match_features_uses_only_history():
    features = build_upcoming_match_features(
        history_df=_history_df(),
        match_date="2024-01-05",
        blue_team="AAA",
        red_team="BBB",
        event="Main",
        patch="1.0",
        season=1,
    )

    assert len(features) == 1
    assert features.loc[0, "blue_team"] == "AAA"
    assert features.loc[0, "red_team"] == "BBB"
    assert pd.isna(features.loc[0, "blue_team_win"])
    assert features.loc[0, "matches_played_diff"] > 0


def test_predict_match_result_returns_probabilities_and_winner():
    result = predict_match_result(
        history_df=_history_df(),
        match_date="2024-01-05",
        blue_team="AAA",
        red_team="BBB",
        event="Main",
        patch="1.0",
        season=1,
    )

    assert 0.0 <= result["blue_win_prob"] <= 1.0
    assert 0.0 <= result["red_win_prob"] <= 1.0
    assert abs((result["blue_win_prob"] + result["red_win_prob"]) - 1.0) < 1e-9
    assert result["predicted_winner"] in {"AAA", "BBB"}
