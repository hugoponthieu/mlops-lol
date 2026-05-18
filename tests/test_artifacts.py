from pathlib import Path

import pandas as pd


def test_clean_matches_artifact_has_required_columns(tmp_path: Path):
    artifact_path = tmp_path / "clean_matches.csv"
    pd.DataFrame(
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
    ).to_csv(artifact_path, index=False)

    loaded = pd.read_csv(artifact_path)

    assert artifact_path.exists()
    assert list(loaded.columns) == [
        "season",
        "date",
        "event",
        "patch",
        "blue_team",
        "red_team",
        "winner",
        "blue_team_win",
    ]
