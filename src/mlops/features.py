from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _safe_mean(values: list[int], default: float = 0.5) -> float:
    return float(np.mean(values)) if values else default


def _coerce_blue_team_win(row: pd.Series) -> int | None:
    value = row.get("blue_team_win", pd.NA)
    if not pd.isna(value):
        return int(value)

    winner = row.get("winner", pd.NA)
    blue_team = row.get("blue_team", pd.NA)
    red_team = row.get("red_team", pd.NA)

    if pd.isna(winner) or pd.isna(blue_team) or pd.isna(red_team):
        return None
    if winner == blue_team:
        return 1
    if winner == red_team:
        return 0
    return None


def build_match_features(
    df: pd.DataFrame,
    base_elo: float = 1500.0,
    k_factor: float = 32.0,
) -> pd.DataFrame:
    ordered = df.sort_values(["date"], kind="mergesort").reset_index(drop=True).copy()

    elo = defaultdict(lambda: base_elo)
    matches_played = defaultdict(int)
    overall_history = defaultdict(list)
    blue_side_history = defaultdict(list)
    red_side_history = defaultdict(list)
    last_played = {}
    h2h = defaultdict(list)
    rows = []

    for _, row in ordered.iterrows():
        blue = row["blue_team"]
        red = row["red_team"]
        match_date = row["date"]

        blue_recent_5 = _safe_mean(overall_history[blue][-5:])
        red_recent_5 = _safe_mean(overall_history[red][-5:])
        blue_recent_10 = _safe_mean(overall_history[blue][-10:])
        red_recent_10 = _safe_mean(overall_history[red][-10:])
        blue_recent_20 = _safe_mean(overall_history[blue][-20:])
        red_recent_20 = _safe_mean(overall_history[red][-20:])

        blue_rest = (match_date - last_played[blue]).days if blue in last_played else -1
        red_rest = (match_date - last_played[red]).days if red in last_played else -1

        h2h_key = tuple(sorted((blue, red)))
        blue_h2h = h2h[h2h_key]
        if blue_h2h:
            blue_h2h_rate = sum(1 for winner in blue_h2h if winner == blue) / len(blue_h2h)
        else:
            blue_h2h_rate = 0.5

        rows.append(
            {
                "date": match_date,
                "blue_team": blue,
                "red_team": red,
                "season": row.get("season", pd.NA),
                "event": row.get("event", pd.NA),
                "patch": row.get("patch", pd.NA),
                "blue_team_win": row.get("blue_team_win", pd.NA),
                "elo_diff": elo[blue] - elo[red],
                "winrate_last_5_diff": blue_recent_5 - red_recent_5,
                "winrate_last_10_diff": blue_recent_10 - red_recent_10,
                "winrate_last_20_diff": blue_recent_20 - red_recent_20,
                "matches_played_diff": matches_played[blue] - matches_played[red],
                "days_since_last_match_diff": blue_rest - red_rest,
                "head_to_head_winrate_diff": blue_h2h_rate - 0.5,
                "blue_side_team_winrate": _safe_mean(blue_side_history[blue]),
                "red_side_team_winrate": _safe_mean(red_side_history[red]),
            }
        )

        actual_blue = _coerce_blue_team_win(row)
        if actual_blue is None:
            continue

        expected_blue = _expected_score(elo[blue], elo[red])
        elo[blue] += k_factor * (actual_blue - expected_blue)
        elo[red] += k_factor * ((1 - actual_blue) - (1 - expected_blue))
        matches_played[blue] += 1
        matches_played[red] += 1
        overall_history[blue].append(actual_blue)
        overall_history[red].append(1 - actual_blue)
        blue_side_history[blue].append(actual_blue)
        red_side_history[red].append(1 - actual_blue)
        last_played[blue] = match_date
        last_played[red] = match_date
        h2h[h2h_key].append(blue if actual_blue == 1 else red)

    return pd.DataFrame(rows)
