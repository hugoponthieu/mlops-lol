from kfp import dsl
from kfp.dsl import Dataset, Input, Output

def build_feature_state(df, base_elo=1500):
    """Build initial state from historical matches"""
    import pandas as pd  # type: ignore

    state = {
        "elo": {},
        "matches_played": {},
        "win_history": {},
        "h2h": {},
    }
    # Initialize Elo
    teams = pd.concat([df["team_1"], df["team_2"]]).unique()
    for team in teams:
        state["elo"][team] = base_elo
        state["matches_played"][team] = 0
        state["win_history"][team] = []
    return state

def compute_features_for_match(match, state):
    """Compute feature vector for a single match"""
    import numpy as np  # type: ignore

    team = match["team_1"]
    opp = match["team_2"]

    # Elo diff
    elo_team = state["elo"].get(team, 1500)
    elo_opp = state["elo"].get(opp, 1500)
    elo_diff = elo_team - elo_opp

    # Rolling winrates
    winrate_10 = (
        np.mean(state["win_history"].get(team, [])[-10:])
        if len(state["win_history"].get(team, [])) > 0
        else 0
    )
    winrate_30 = (
        np.mean(state["win_history"].get(team, [])[-30:])
        if len(state["win_history"].get(team, [])) > 0
        else 0
    )
    winrate_10_diff = winrate_10 - (
        np.mean(state["win_history"].get(opp, [])[-10:])
        if len(state["win_history"].get(opp, [])) > 0
        else 0
    )
    winrate_30_diff = winrate_30 - (
        np.mean(state["win_history"].get(opp, [])[-30:])
        if len(state["win_history"].get(opp, [])) > 0
        else 0
    )

    # Experience / matches played
    experience_diff = state["matches_played"].get(team, 0) - state[
        "matches_played"
    ].get(opp, 0)

    # Rank diff
    rank_diff = match["rank_1"] - match["rank_2"]

    # H2H winrate
    h2h_key = (team, opp)
    h2h_list = state["h2h"].get(h2h_key, [])
    h2h_winrate = np.mean(h2h_list) if len(h2h_list) > 0 else 0.5

    return {
        "elo_diff": elo_diff,
        "winrate_10_diff": winrate_10_diff,
        "winrate_30_diff": winrate_30_diff,
        "experience_diff": experience_diff,
        "rank_diff": rank_diff,
        "h2h_winrate": h2h_winrate,
    }

def update_state_with_result(match, state, k=32):
    """Update Elo, rolling winrates, H2H after a match"""
    team = match["team_1"]
    opp = match["team_2"]
    result = int(match["match_winner"] == 1)

    # Update Elo
    r_team = state["elo"].get(team, 1500)
    r_opp = state["elo"].get(opp, 1500)
    exp = 1 / (1 + 10 ** ((r_opp - r_team) / 400))
    state["elo"][team] = r_team + k * (result - exp)
    state["elo"][opp] = r_opp + k * ((1 - result) - (1 - exp))

    # Update win history
    state["win_history"].setdefault(team, []).append(result)
    state["win_history"].setdefault(opp, []).append(1 - result)

    # Update matches played
    state["matches_played"][team] = state["matches_played"].get(team, 0) + 1
    state["matches_played"][opp] = state["matches_played"].get(opp, 0) + 1

    # Update H2H
    h2h_key = (team, opp)
    state["h2h"].setdefault(h2h_key, []).append(result)
    return state

def match_to_features(preprocessor, history_df, match):
    """
    match: dict with team_1, team_2, rank_1, rank_2, date
    """
    import pandas as pd  # type: ignore

    # Build state from past only
    past = history_df[history_df["date"] < match["date"]]
    state = build_feature_state(past)

    # Replay past matches to update state
    for _, m in past.sort_values("date").iterrows():
        update_state_with_result(m, state)

    # Compute features for the future match
    X = pd.DataFrame([compute_features_for_match(match, state)])

    X_scaled = preprocessor.transform(X)

    return X_scaled

@dsl.component(
    packages_to_install=[
        "pandas==3.0.2",
        "tensorflow==2.21.0",
        "scikit-learn==1.8.0",
        "joblib==1.5.3",
        "numpy==2.4.4",
        "requests==2.33.1", # CVE-2026-25645
        "pygments==2.20.0" # CVE-2026-4539
    ],
    base_image="python:3.12",
)
def feature_evaluate_model(
    matches_csv: Input[Dataset],
    preprocessed_test_csv: Input[Dataset],
    model_path: Input[Dataset],
    preprocessor_path: Input[Dataset],
):
    import pandas as pd

    df = pd.read_csv(matches_csv.path)
    df["date"] = pd.to_datetime(df["date"])

    df_test = pd.read_csv(preprocessed_test_csv.path)

    from keras.models import load_model
    model = load_model(model_path.path)

    import joblib
    preprocessor = joblib.load(preprocessor_path.path)

    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
    target = "team_1_wins"

    X_test = df_test.drop(target, axis=1)
    y_test = df_test[target]

    y_proba = model.predict(X_test).ravel()  # type: ignore
    y_pred = (y_proba >= 0.5).astype(int)

    df_test_predicted = pd.concat([df_test, pd.Series(y_pred, name="prediction")], axis=1)

    # Do a test prediction on a made up match
    future_match = {
        "team_1": "Rugratz",
        "team_2": "Bad News Bears",
        "rank_1": 61,
        "rank_2": 38,
        "date": pd.Timestamp("2024-06-01"),
    }

    X = match_to_features(preprocessor, df, future_match)
    p = model.predict(X)[0][0]  # type: ignore

    print(f"{future_match['team_1']} win probability: {p:.2%}")
    print(
        "Predicted winner:", future_match["team_1"] if p > 0.5 else future_match["team_2"]
    )