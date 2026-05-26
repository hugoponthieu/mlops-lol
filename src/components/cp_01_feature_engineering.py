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

@dsl.component(
    packages_to_install=[
        "pandas==3.0.2",
        "numpy==2.4.4",
    ],
    base_image="python:3.14",
    additional_funcs=[build_feature_state, compute_features_for_match, update_state_with_result],
)
def feature_engineering(
    matches_csv: Input[Dataset],
    features_csv: Output[Dataset],
):
    import pandas as pd  # type: ignore

    matches = pd.read_csv(matches_csv.path)
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values("date").reset_index(drop=True)

    # Build initial state
    state = build_feature_state(matches)
    X = []
    y = []

    # Compute features for each match
    for _, match in matches.iterrows():
        features = compute_features_for_match(match, state)
        X.append(features)
        y.append(int(match["match_winner"] == 1))
        state = update_state_with_result(match, state)

    df_featured = pd.concat([matches, pd.DataFrame(X), pd.Series(y, name="team_1_wins")], axis=1)

    df_featured.to_csv(features_csv.path, index=False)