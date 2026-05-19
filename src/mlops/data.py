from __future__ import annotations

from pathlib import Path
from typing import IO, Union

import pandas as pd

DataSource = Union[str, Path, IO[str]]
REQUIRED_COLUMNS = [
    "season",
    "date",
    "event",
    "patch",
    "blue_team",
    "red_team",
    "winner",
]


def _strip_preserve_nulls(series: pd.Series) -> pd.Series:
    stripped = series.where(series.notna(), pd.NA).astype("string").str.strip()
    return stripped.mask(stripped == "", pd.NA)


def load_matches(source: DataSource) -> pd.DataFrame:
    df = pd.read_csv(source)
    stripped_columns = [col.strip() for col in df.columns]
    if len(set(stripped_columns)) != len(stripped_columns):
        raise ValueError("header names must be unique after trimming whitespace")
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in stripped_columns]
    if missing_columns:
        raise ValueError(
            f"missing required columns after trimming headers: {', '.join(missing_columns)}"
        )
    df.columns = stripped_columns
    df["date"] = _strip_preserve_nulls(df["date"])
    try:
        df["date"] = pd.to_datetime(df["date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("date values must be valid datetimes") from exc
    if df["date"].isna().any():
        raise ValueError("date values must be present")
    df["patch"] = _strip_preserve_nulls(df["patch"])
    df["event"] = _strip_preserve_nulls(df["event"])
    df["blue_team"] = _strip_preserve_nulls(df["blue_team"])
    df["red_team"] = _strip_preserve_nulls(df["red_team"])
    df["winner"] = _strip_preserve_nulls(df["winner"])
    winner_matches_blue = (
        df["winner"].notna()
        & df["blue_team"].notna()
        & (df["winner"] == df["blue_team"])
    )
    winner_matches_red = (
        df["winner"].notna()
        & df["red_team"].notna()
        & (df["winner"] == df["red_team"])
    )
    invalid_winner_mask = df["winner"].notna() & ~(winner_matches_blue | winner_matches_red)
    if invalid_winner_mask.any():
        raise ValueError("winner must match either blue_team or red_team when present")
    blue_team_win = pd.Series(pd.NA, index=df.index, dtype="Int64")
    blue_team_win.loc[winner_matches_blue] = 1
    blue_team_win.loc[winner_matches_red & ~winner_matches_blue] = 0
    df["blue_team_win"] = blue_team_win
    df = df.sort_values(["date"], kind="mergesort").reset_index(drop=True)
    leading_columns = [
        "season",
        "date",
        "event",
        "patch",
        "blue_team",
        "red_team",
        "winner",
    ]
    remaining_columns = [
        col
        for col in df.columns
        if col not in leading_columns and col != "blue_team_win"
    ]
    return df[leading_columns + remaining_columns + ["blue_team_win"]]
