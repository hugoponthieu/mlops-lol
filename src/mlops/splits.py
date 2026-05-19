from __future__ import annotations

import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    valid_frac: float = 0.15,
):
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")
    if not 0 < valid_frac < 1:
        raise ValueError("valid_frac must be between 0 and 1")
    if train_frac + valid_frac >= 1:
        raise ValueError("train_frac + valid_frac must be less than 1")

    ordered = df.sort_values(["date"], kind="mergesort").reset_index(drop=True)
    split_days = ordered["date"].dt.normalize()
    unique_days = split_days.drop_duplicates().reset_index(drop=True)

    train_end = int(len(unique_days) * train_frac)
    valid_end = int(len(unique_days) * (train_frac + valid_frac))

    train_days = unique_days.iloc[:train_end]
    valid_days = unique_days.iloc[train_end:valid_end]
    test_days = unique_days.iloc[valid_end:]

    train_df = ordered[split_days.isin(train_days)].reset_index(drop=True)
    valid_df = ordered[split_days.isin(valid_days)].reset_index(drop=True)
    test_df = ordered[split_days.isin(test_days)].reset_index(drop=True)
    return train_df, valid_df, test_df


def split_features_and_target(df: pd.DataFrame):
    drop_columns = ["date", "blue_team", "red_team", "blue_team_win"]
    X = df.drop(columns=drop_columns)
    y = df["blue_team_win"].copy()
    return X, y
