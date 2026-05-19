from io import StringIO

import pandas as pd
import pytest

from mlops.data import load_matches


def test_load_matches_parses_dates_and_builds_target():
    csv_text = """ season , date , event , patch , blue_team , red_team , winner 
1,2011-06-20,Main,,AAA,BBB,AAA
1,2011-06-21,Main,11.1,CCC,DDD,DDD
"""
    df = load_matches(StringIO(csv_text))

    assert list(df["blue_team"]) == ["AAA", "CCC"]
    assert str(df["date"].dtype).startswith("datetime64")
    assert list(df["blue_team_win"]) == [1, 0]
    assert str(df["blue_team_win"].dtype) == "Int64"
    assert list(df.columns[:7]) == [
        "season",
        "date",
        "event",
        "patch",
        "blue_team",
        "red_team",
        "winner",
    ]


def test_load_matches_preserves_nulls_and_keeps_duplicate_date_order():
    csv_text = """season,date,event,patch,blue_team,red_team,winner
1,2011-06-21,Main,,AAA,BBB,AAA
1,2011-06-20,,11.1,CCC,DDD,
1,2011-06-21,Main,11.2,,EEE,
"""
    df = load_matches(StringIO(csv_text))

    assert list(df["blue_team"].fillna("<missing>")) == ["CCC", "AAA", "<missing>"]
    assert list(df["winner"].fillna("<missing>")) == ["<missing>", "AAA", "<missing>"]
    assert df["blue_team_win"].isna().tolist() == [True, False, True]
    assert df["blue_team_win"].dropna().tolist() == [1]
    assert list(df["patch"].fillna("<missing>")) == ["11.1", "<missing>", "11.2"]
    assert list(df["event"].fillna("<missing>")) == ["<missing>", "Main", "Main"]


def test_load_matches_treats_whitespace_only_cells_as_missing():
    csv_text = """season,date,event,patch,blue_team,red_team,winner
1,2011-06-20,Main,  ,   ,BBB,   
"""
    df = load_matches(StringIO(csv_text))

    assert df["patch"].isna().tolist() == [True]
    assert df["blue_team"].isna().tolist() == [True]
    assert df["winner"].isna().tolist() == [True]
    assert df["blue_team_win"].isna().tolist() == [True]
    assert str(df["blue_team_win"].dtype) == "Int64"


def test_load_matches_rejects_blank_dates():
    csv_text = """season,date,event,patch,blue_team,red_team,winner
1,,Main,,AAA,BBB,AAA
"""

    with pytest.raises(ValueError, match="date values must be present"):
        load_matches(StringIO(csv_text))


def test_load_matches_rejects_invalid_dates():
    csv_text = """season,date,event,patch,blue_team,red_team,winner
1,not-a-date,Main,,AAA,BBB,AAA
"""

    with pytest.raises(ValueError, match="date values must be valid datetimes"):
        load_matches(StringIO(csv_text))


def test_load_matches_rejects_duplicate_headers_after_trimming():
    csv_text = """season,date,event,patch,blue_team,red_team,winner , winner
1,2011-06-20,Main,,AAA,BBB,AAA,BBB
"""

    with pytest.raises(ValueError, match="unique after trimming whitespace"):
        load_matches(StringIO(csv_text))


def test_load_matches_rejects_missing_required_columns():
    csv_text = """season,date,event,patch,blue_team,red_team
1,2011-06-20,Main,,AAA,BBB
"""

    with pytest.raises(ValueError, match="missing required columns"):
        load_matches(StringIO(csv_text))


def test_load_matches_rejects_winner_values_that_match_neither_side():
    csv_text = """season,date,event,patch,blue_team,red_team,winner
1,2011-06-20,Main,,AAA,BBB,CCC
"""

    with pytest.raises(ValueError, match="winner must match either blue_team or red_team"):
        load_matches(StringIO(csv_text))


def test_load_matches_keeps_known_columns_first_extra_columns_middle_and_target_last():
    csv_text = """season,date,event,patch,blue_team,red_team,winner,venue,match_id
1,2011-06-20,Main,,AAA,BBB,AAA,Studio,42
"""
    df = load_matches(StringIO(csv_text))

    assert list(df.columns) == [
        "season",
        "date",
        "event",
        "patch",
        "blue_team",
        "red_team",
        "winner",
        "venue",
        "match_id",
        "blue_team_win",
    ]
