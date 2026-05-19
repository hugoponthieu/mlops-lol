from __future__ import annotations

import pandas as pd

from mlops.features import build_match_features
from mlops.modeling import train_logistic_regression
from mlops.splits import split_features_and_target


def build_upcoming_match_features(
    history_df: pd.DataFrame,
    match_date: str,
    blue_team: str,
    red_team: str,
    event: str = "Main",
    patch: str = "",
    season: int | None = None,
) -> pd.DataFrame:
    future_match = pd.DataFrame(
        [
            {
                "season": season if season is not None else pd.NA,
                "date": pd.to_datetime(match_date),
                "event": event,
                "patch": patch,
                "blue_team": blue_team,
                "red_team": red_team,
                "winner": pd.NA,
                "blue_team_win": pd.NA,
            }
        ]
    )
    combined = pd.concat([history_df.copy(), future_match], ignore_index=True, sort=False)
    feature_df = build_match_features(combined)
    return feature_df.tail(1).reset_index(drop=True)


def predict_match_result(
    history_df: pd.DataFrame,
    match_date: str,
    blue_team: str,
    red_team: str,
    event: str = "Main",
    patch: str = "",
    season: int | None = None,
) -> dict[str, object]:
    history_features = build_match_features(history_df)
    labeled_history = history_features[history_features["blue_team_win"].notna()].copy()
    labeled_history["blue_team_win"] = labeled_history["blue_team_win"].astype("Int64")

    X_train, y_train = split_features_and_target(labeled_history)
    model = train_logistic_regression(X_train, y_train)

    upcoming_features = build_upcoming_match_features(
        history_df=history_df,
        match_date=match_date,
        blue_team=blue_team,
        red_team=red_team,
        event=event,
        patch=patch,
        season=season,
    )
    X_match, _ = split_features_and_target(upcoming_features)
    blue_win_prob = float(model.predict_proba(X_match)[0, 1])
    red_win_prob = 1.0 - blue_win_prob

    return {
        "date": pd.to_datetime(match_date),
        "blue_team": blue_team,
        "red_team": red_team,
        "event": event,
        "patch": patch,
        "blue_win_prob": blue_win_prob,
        "red_win_prob": red_win_prob,
        "predicted_winner": blue_team if blue_win_prob >= 0.5 else red_team,
    }
