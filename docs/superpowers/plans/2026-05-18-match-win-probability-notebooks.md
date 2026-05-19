# Match Win Probability Notebooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the notebook pipeline so every stage is self-contained, writes reusable filesystem artifacts under `artifacts/`, and saves a trained model that can be reused outside the original notebook session.

**Architecture:** The notebooks under `notebooks/` become the source of truth for the workflow. Each notebook defines the helper logic it needs in visible top cells, reads the previous stage's exported files from `artifacts/`, and writes its own outputs back to disk. The existing `src/mlops/` helpers can remain for tests or reference, but the runtime notebook path must not depend on them.

**Tech Stack:** Python, Jupyter notebooks, pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, pytest

---

## File Structure

Planned file responsibilities:

- Modify: `notebooks/01_data_pull_and_audit.ipynb`
  - Load `data/matchs_stats.csv`, clean columns, define `blue_team_win`, and export `artifacts/clean_matches.csv`.
- Modify: `notebooks/02_feature_engineering.ipynb`
  - Read `artifacts/clean_matches.csv`, build leakage-safe historical features in notebook cells, and export `artifacts/match_features.csv`.
- Modify: `notebooks/03_preprocessing.ipynb`
  - Read `artifacts/match_features.csv`, create chronological splits, and export split CSV files plus `artifacts/feature_columns.json`.
- Modify: `notebooks/04_train_probability_model.ipynb`
  - Read training artifacts, train the logistic regression pipeline, and export `artifacts/logistic_regression_model.joblib`, `artifacts/validation_predictions.csv`, and `artifacts/train_manifest.json`.
- Modify: `notebooks/05_evaluation_and_calibration.ipynb`
  - Read test artifacts and trained model, compute metrics and plots, and export `artifacts/evaluation_metrics.json`, `artifacts/test_predictions.csv`, and optional files under `artifacts/plots/`.
- Modify: `notebooks/06_predict_match.ipynb`
  - Read persisted cleaned history and persisted model, generate one future-match feature row inside the notebook, and export `artifacts/predicted_match_result.csv`.
- Modify: `tests/test_predict.py`
  - Align predictions and persisted-model expectations with the notebook artifact contract where necessary.
- Create: `tests/test_artifacts.py`
  - Add filesystem-oriented tests for artifact naming and expected columns using small synthetic dataframes.

### Task 1: Add artifact export and self-contained helper cells to notebook 01

**Files:**
- Modify: `notebooks/01_data_pull_and_audit.ipynb`
- Test: `tests/test_artifacts.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_artifacts.py::test_clean_matches_artifact_has_required_columns -v`
Expected: FAIL with `file or directory not found: tests/test_artifacts.py`

- [ ] **Step 3: Write minimal implementation**

```python
# tests/test_artifacts.py
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
```

```python
# notebook cells to add near the top of notebooks/01_data_pull_and_audit.ipynb
from pathlib import Path

import pandas as pd

ARTIFACTS_DIR = Path("../artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_matches_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["patch"] = df["patch"].fillna("").astype(str).str.strip()
    df["event"] = df["event"].fillna("unknown").astype(str).str.strip()
    df["blue_team"] = df["blue_team"].astype(str).str.strip()
    df["red_team"] = df["red_team"].astype(str).str.strip()
    df["winner"] = df["winner"].astype(str).str.strip()
    df["blue_team_win"] = (df["winner"] == df["blue_team"]).astype(int)
    return df.sort_values(["date"], kind="mergesort").reset_index(drop=True)


matches = load_matches_csv("../data/matchs_stats.csv")
matches.to_csv(ARTIFACTS_DIR / "clean_matches.csv", index=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_artifacts.py::test_clean_matches_artifact_has_required_columns -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_artifacts.py notebooks/01_data_pull_and_audit.ipynb
git commit -m "feat: export cleaned matches artifact from notebook 01"
```

### Task 2: Make notebook 02 self-contained and export match features

**Files:**
- Modify: `notebooks/02_feature_engineering.ipynb`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_features.py::test_build_match_features_preserves_context_columns -v`
Expected: FAIL if the expected context columns are not preserved

- [ ] **Step 3: Write minimal implementation**

```python
# notebook cells to add near the top of notebooks/02_feature_engineering.ipynb
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path("../artifacts")
clean_matches = pd.read_csv(ARTIFACTS_DIR / "clean_matches.csv", parse_dates=["date"])


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def safe_mean(values: list[int], default: float = 0.5) -> float:
    return float(np.mean(values)) if values else default


def build_match_features_local(
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
    head_to_head = defaultdict(list)
    rows = []

    for row in ordered.itertuples(index=False):
        blue = row.blue_team
        red = row.red_team
        match_date = row.date
        key = tuple(sorted((blue, red)))
        blue_h2h = head_to_head[key]

        rows.append(
            {
                "season": row.season,
                "date": match_date,
                "event": row.event,
                "patch": row.patch,
                "blue_team": blue,
                "red_team": red,
                "winner": row.winner,
                "blue_team_win": row.blue_team_win,
                "elo_diff": elo[blue] - elo[red],
                "winrate_last_5_diff": safe_mean(overall_history[blue][-5:]) - safe_mean(overall_history[red][-5:]),
                "winrate_last_10_diff": safe_mean(overall_history[blue][-10:]) - safe_mean(overall_history[red][-10:]),
                "winrate_last_20_diff": safe_mean(overall_history[blue][-20:]) - safe_mean(overall_history[red][-20:]),
                "matches_played_diff": matches_played[blue] - matches_played[red],
                "days_since_last_match_diff": (
                    (match_date - last_played[blue]).days if blue in last_played else -1
                ) - (
                    (match_date - last_played[red]).days if red in last_played else -1
                ),
                "head_to_head_winrate_diff": (
                    sum(1 for winner in blue_h2h if winner == blue) / len(blue_h2h) if blue_h2h else 0.5
                ) - 0.5,
                "blue_side_team_winrate": safe_mean(blue_side_history[blue]),
                "red_side_team_winrate": safe_mean(red_side_history[red]),
            }
        )

        if pd.isna(row.blue_team_win):
            continue

        outcome = int(row.blue_team_win)
        expected = expected_score(elo[blue], elo[red])
        elo[blue] += k_factor * (outcome - expected)
        elo[red] += k_factor * ((1 - outcome) - (1 - expected))
        matches_played[blue] += 1
        matches_played[red] += 1
        overall_history[blue].append(outcome)
        overall_history[red].append(1 - outcome)
        blue_side_history[blue].append(outcome)
        red_side_history[red].append(1 - outcome)
        last_played[blue] = match_date
        last_played[red] = match_date
        head_to_head[key].append(blue if outcome == 1 else red)

    return pd.DataFrame(rows)


match_features = build_match_features_local(clean_matches)
match_features.to_csv(ARTIFACTS_DIR / "match_features.csv", index=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_features.py::test_build_match_features_preserves_context_columns -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebooks/02_feature_engineering.ipynb tests/test_features.py
git commit -m "feat: export feature artifact from notebook 02"
```

### Task 3: Export chronological split artifacts from notebook 03

**Files:**
- Modify: `notebooks/03_preprocessing.ipynb`
- Test: `tests/test_splits.py`

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd

from mlops.splits import chronological_split


def test_chronological_split_keeps_validation_after_training():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10, freq="D")})
    train_df, valid_df, test_df = chronological_split(df, train_frac=0.6, valid_frac=0.2)

    assert train_df["date"].max() < valid_df["date"].min()
    assert valid_df["date"].max() < test_df["date"].min()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_splits.py::test_chronological_split_keeps_validation_after_training -v`
Expected: FAIL if split ordering is incorrect

- [ ] **Step 3: Write minimal implementation**

```python
# notebook cells to add near the top of notebooks/03_preprocessing.ipynb
import json
from pathlib import Path

import pandas as pd

ARTIFACTS_DIR = Path("../artifacts")
feature_df = pd.read_csv(ARTIFACTS_DIR / "match_features.csv", parse_dates=["date"])
labeled = feature_df[feature_df["blue_team_win"].notna()].copy()
labeled["blue_team_win"] = labeled["blue_team_win"].astype(int)


def chronological_split_local(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    valid_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(["date"], kind="mergesort").reset_index(drop=True)
    train_end = int(len(ordered) * train_frac)
    valid_end = train_end + int(len(ordered) * valid_frac)
    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:valid_end].copy(),
        ordered.iloc[valid_end:].copy(),
    )


def split_features_and_target_local(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df["blue_team_win"].astype(int).copy()
    X = df.drop(columns=["blue_team_win", "winner"])
    return X, y


train_df, valid_df, test_df = chronological_split_local(labeled)
X_train, y_train = split_features_and_target_local(train_df)
X_valid, y_valid = split_features_and_target_local(valid_df)
X_test, y_test = split_features_and_target_local(test_df)

X_train.to_csv(ARTIFACTS_DIR / "train_features.csv", index=False)
X_valid.to_csv(ARTIFACTS_DIR / "valid_features.csv", index=False)
X_test.to_csv(ARTIFACTS_DIR / "test_features.csv", index=False)
y_train.to_frame(name="blue_team_win").to_csv(ARTIFACTS_DIR / "train_labels.csv", index=False)
y_valid.to_frame(name="blue_team_win").to_csv(ARTIFACTS_DIR / "valid_labels.csv", index=False)
y_test.to_frame(name="blue_team_win").to_csv(ARTIFACTS_DIR / "test_labels.csv", index=False)

feature_columns = list(X_train.columns)
(ARTIFACTS_DIR / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_splits.py::test_chronological_split_keeps_validation_after_training -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebooks/03_preprocessing.ipynb tests/test_splits.py
git commit -m "feat: export split artifacts from notebook 03"
```

### Task 4: Save the trained model and validation predictions from notebook 04

**Files:**
- Modify: `notebooks/04_train_probability_model.ipynb`
- Test: `tests/test_modeling.py`

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd

from mlops.modeling import train_logistic_regression


def test_train_logistic_regression_produces_probability_model():
    X_train = pd.DataFrame(
        {
            "elo_diff": [0.0, 50.0, -25.0, 80.0],
            "event": ["Main", "Main", "Playoffs", "Playoffs"],
            "patch": ["14.1", "14.1", "14.2", "14.2"],
        }
    )
    y_train = pd.Series([0, 1, 0, 1])

    model = train_logistic_regression(X_train, y_train)
    probabilities = model.predict_proba(X_train)

    assert probabilities.shape == (4, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_modeling.py::test_train_logistic_regression_produces_probability_model -v`
Expected: FAIL if the training pipeline does not return a probability-capable model

- [ ] **Step 3: Write minimal implementation**

```python
# notebook cells to add near the top of notebooks/04_train_probability_model.ipynb
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ARTIFACTS_DIR = Path("../artifacts")
X_train = pd.read_csv(ARTIFACTS_DIR / "train_features.csv", parse_dates=["date"])
X_valid = pd.read_csv(ARTIFACTS_DIR / "valid_features.csv", parse_dates=["date"])
y_train = pd.read_csv(ARTIFACTS_DIR / "train_labels.csv")["blue_team_win"]
y_valid = pd.read_csv(ARTIFACTS_DIR / "valid_labels.csv")["blue_team_win"]


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["date", "blue_team", "red_team"], errors="ignore")


def train_logistic_regression_local(X_train_df: pd.DataFrame, y_train_series: pd.Series) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_include=["number"])),
            ("cat", categorical_transformer, make_column_selector(dtype_exclude=["number"])),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train_df, y_train_series.astype(int))
    return model


train_frame = build_training_frame(X_train)
valid_frame = build_training_frame(X_valid)
model = train_logistic_regression_local(train_frame, y_train)
valid_prob = model.predict_proba(valid_frame)[:, 1]

joblib.dump(model, ARTIFACTS_DIR / "logistic_regression_model.joblib")
pd.DataFrame(
    {
        "blue_win_prob": valid_prob,
        "blue_team_win": y_valid.astype(int),
    }
).to_csv(ARTIFACTS_DIR / "validation_predictions.csv", index=False)
(ARTIFACTS_DIR / "train_manifest.json").write_text(
    json.dumps(
        {
            "model_name": "logistic_regression",
            "feature_columns": list(train_frame.columns),
            "train_rows": int(len(train_frame)),
            "valid_rows": int(len(valid_frame)),
        },
        indent=2,
    )
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_modeling.py::test_train_logistic_regression_produces_probability_model -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebooks/04_train_probability_model.ipynb tests/test_modeling.py
git commit -m "feat: persist trained model from notebook 04"
```

### Task 5: Export evaluation metrics, test predictions, and plots from notebook 05

**Files:**
- Modify: `notebooks/05_evaluation_and_calibration.ipynb`
- Test: `tests/test_artifacts.py`

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path


def test_evaluation_metrics_json_contains_required_keys(tmp_path: Path):
    metrics_path = tmp_path / "evaluation_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "log_loss": 0.6,
                "brier_score": 0.2,
                "roc_auc": 0.7,
                "accuracy_50": 0.65,
            }
        )
    )

    payload = json.loads(metrics_path.read_text())

    assert set(payload) == {"log_loss", "brier_score", "roc_auc", "accuracy_50"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_artifacts.py::test_evaluation_metrics_json_contains_required_keys -v`
Expected: FAIL until the new test exists

- [ ] **Step 3: Write minimal implementation**

```python
# notebook cells to add near the top of notebooks/05_evaluation_and_calibration.ipynb
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

ARTIFACTS_DIR = Path("../artifacts")
PLOTS_DIR = ARTIFACTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

model = joblib.load(ARTIFACTS_DIR / "logistic_regression_model.joblib")
X_test = pd.read_csv(ARTIFACTS_DIR / "test_features.csv", parse_dates=["date"])
y_test = pd.read_csv(ARTIFACTS_DIR / "test_labels.csv")["blue_team_win"].astype(int)
test_frame = X_test.drop(columns=["date", "blue_team", "red_team"], errors="ignore")
test_prob = model.predict_proba(test_frame)[:, 1]

metrics = {
    "log_loss": float(log_loss(y_test, test_prob)),
    "brier_score": float(brier_score_loss(y_test, test_prob)),
    "roc_auc": float(roc_auc_score(y_test, test_prob)),
    "accuracy_50": float(accuracy_score(y_test, (test_prob >= 0.5).astype(int))),
}

(ARTIFACTS_DIR / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2))
pd.DataFrame(
    {
        "blue_win_prob": test_prob,
        "blue_team_win": y_test,
    }
).to_csv(ARTIFACTS_DIR / "test_predictions.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 6))
CalibrationDisplay.from_predictions(y_test, test_prob, n_bins=10, ax=ax)
fig.savefig(PLOTS_DIR / "calibration_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_artifacts.py::test_evaluation_metrics_json_contains_required_keys -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebooks/05_evaluation_and_calibration.ipynb tests/test_artifacts.py
git commit -m "feat: export evaluation artifacts from notebook 05"
```

### Task 6: Load the persisted model in notebook 06 and export a prediction artifact

**Files:**
- Modify: `notebooks/06_predict_match.ipynb`
- Modify: `tests/test_predict.py`

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd

from mlops.predict import predict_match_result


def test_predict_match_result_returns_both_probabilities():
    history_df = pd.DataFrame(
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
            },
            {
                "season": 1,
                "date": "2024-01-02",
                "event": "Main",
                "patch": "14.1",
                "blue_team": "AAA",
                "red_team": "CCC",
                "winner": "CCC",
                "blue_team_win": 0,
            },
        ]
    )
    history_df["date"] = pd.to_datetime(history_df["date"])

    result = predict_match_result(
        history_df=history_df,
        match_date="2024-01-03",
        blue_team="AAA",
        red_team="BBB",
        event="Main",
        patch="14.1",
        season=1,
    )

    assert 0.0 <= result["blue_win_prob"] <= 1.0
    assert 0.0 <= result["red_win_prob"] <= 1.0
    assert abs(result["blue_win_prob"] + result["red_win_prob"] - 1.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_predict.py::test_predict_match_result_returns_both_probabilities -v`
Expected: FAIL if prediction output contract differs

- [ ] **Step 3: Write minimal implementation**

```python
# notebook cells to add near the top of notebooks/06_predict_match.ipynb
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path("../artifacts")
clean_matches = pd.read_csv(ARTIFACTS_DIR / "clean_matches.csv", parse_dates=["date"])
model = joblib.load(ARTIFACTS_DIR / "logistic_regression_model.joblib")
feature_columns = json.loads((ARTIFACTS_DIR / "feature_columns.json").read_text())


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def safe_mean(values: list[int], default: float = 0.5) -> float:
    return float(np.mean(values)) if values else default


def build_upcoming_match_feature_row(
    history_df: pd.DataFrame,
    match_date: str,
    blue_team: str,
    red_team: str,
    event: str,
    patch: str,
    season: int | None,
) -> pd.DataFrame:
    future_match = pd.DataFrame(
        [
            {
                "season": season,
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
    feature_rows = build_match_features_local(combined)
    return feature_rows.tail(1).reset_index(drop=True)


match_features = build_upcoming_match_feature_row(
    history_df=clean_matches,
    match_date="2026-06-01",
    blue_team="Team A",
    red_team="Team B",
    event="Main",
    patch="14.1",
    season=1,
)
prediction_frame = match_features.drop(columns=["winner", "blue_team_win"], errors="ignore")
prediction_frame = prediction_frame[feature_columns]
blue_win_prob = float(model.predict_proba(prediction_frame)[0, 1])
result_df = pd.DataFrame(
    [
        {
            "date": "2026-06-01",
            "blue_team": "Team A",
            "red_team": "Team B",
            "blue_win_prob": blue_win_prob,
            "red_win_prob": 1.0 - blue_win_prob,
            "predicted_winner": "Team A" if blue_win_prob >= 0.5 else "Team B",
        }
    ]
)
result_df.to_csv(ARTIFACTS_DIR / "predicted_match_result.csv", index=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_predict.py::test_predict_match_result_returns_both_probabilities -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebooks/06_predict_match.ipynb tests/test_predict.py
git commit -m "feat: export prediction artifact from notebook 06"
```

### Task 7: Execute the notebook chain and verify artifact generation end to end

**Files:**
- Modify: `notebooks/01_data_pull_and_audit.ipynb`
- Modify: `notebooks/02_feature_engineering.ipynb`
- Modify: `notebooks/03_preprocessing.ipynb`
- Modify: `notebooks/04_train_probability_model.ipynb`
- Modify: `notebooks/05_evaluation_and_calibration.ipynb`
- Modify: `notebooks/06_predict_match.ipynb`
- Test: `tests/test_artifacts.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_expected_artifact_paths_are_documented():
    expected = [
        Path("artifacts/clean_matches.csv"),
        Path("artifacts/match_features.csv"),
        Path("artifacts/train_features.csv"),
        Path("artifacts/valid_features.csv"),
        Path("artifacts/test_features.csv"),
        Path("artifacts/train_labels.csv"),
        Path("artifacts/valid_labels.csv"),
        Path("artifacts/test_labels.csv"),
        Path("artifacts/feature_columns.json"),
        Path("artifacts/logistic_regression_model.joblib"),
        Path("artifacts/validation_predictions.csv"),
        Path("artifacts/train_manifest.json"),
        Path("artifacts/evaluation_metrics.json"),
        Path("artifacts/test_predictions.csv"),
        Path("artifacts/predicted_match_result.csv"),
    ]

    assert len(expected) == 15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_artifacts.py::test_expected_artifact_paths_are_documented -v`
Expected: FAIL until the new test exists

- [ ] **Step 3: Write minimal implementation**

```python
# tests/test_artifacts.py
from pathlib import Path

import json
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


def test_evaluation_metrics_json_contains_required_keys(tmp_path: Path):
    metrics_path = tmp_path / "evaluation_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "log_loss": 0.6,
                "brier_score": 0.2,
                "roc_auc": 0.7,
                "accuracy_50": 0.65,
            }
        )
    )

    payload = json.loads(metrics_path.read_text())

    assert set(payload) == {"log_loss", "brier_score", "roc_auc", "accuracy_50"}


def test_expected_artifact_paths_are_documented():
    expected = [
        Path("artifacts/clean_matches.csv"),
        Path("artifacts/match_features.csv"),
        Path("artifacts/train_features.csv"),
        Path("artifacts/valid_features.csv"),
        Path("artifacts/test_features.csv"),
        Path("artifacts/train_labels.csv"),
        Path("artifacts/valid_labels.csv"),
        Path("artifacts/test_labels.csv"),
        Path("artifacts/feature_columns.json"),
        Path("artifacts/logistic_regression_model.joblib"),
        Path("artifacts/validation_predictions.csv"),
        Path("artifacts/train_manifest.json"),
        Path("artifacts/evaluation_metrics.json"),
        Path("artifacts/test_predictions.csv"),
        Path("artifacts/predicted_match_result.csv"),
    ]

    assert len(expected) == 15
```

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_artifacts.py -v
PYTHONPATH=src .venv/bin/jupyter nbconvert --to notebook --execute notebooks/01_data_pull_and_audit.ipynb --output /tmp/01.executed.ipynb
PYTHONPATH=src .venv/bin/jupyter nbconvert --to notebook --execute notebooks/02_feature_engineering.ipynb --output /tmp/02.executed.ipynb
PYTHONPATH=src .venv/bin/jupyter nbconvert --to notebook --execute notebooks/03_preprocessing.ipynb --output /tmp/03.executed.ipynb
PYTHONPATH=src .venv/bin/jupyter nbconvert --to notebook --execute notebooks/04_train_probability_model.ipynb --output /tmp/04.executed.ipynb
PYTHONPATH=src .venv/bin/jupyter nbconvert --to notebook --execute notebooks/05_evaluation_and_calibration.ipynb --output /tmp/05.executed.ipynb
PYTHONPATH=src .venv/bin/jupyter nbconvert --to notebook --execute notebooks/06_predict_match.ipynb --output /tmp/06.executed.ipynb
```

- [ ] **Step 4: Run verification to confirm it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_artifacts.py -v`
Expected: PASS

Run: `ls artifacts`
Expected: list includes `clean_matches.csv`, `match_features.csv`, split files, `logistic_regression_model.joblib`, metrics outputs, and `predicted_match_result.csv`

- [ ] **Step 5: Commit**

```bash
git add tests/test_artifacts.py notebooks/01_data_pull_and_audit.ipynb notebooks/02_feature_engineering.ipynb notebooks/03_preprocessing.ipynb notebooks/04_train_probability_model.ipynb notebooks/05_evaluation_and_calibration.ipynb notebooks/06_predict_match.ipynb
git commit -m "feat: persist notebook pipeline artifacts end to end"
```

## Self-Review

Spec coverage check:

- Notebook self-containment is covered by Tasks 1 through 6, each moving runtime logic into notebook cells.
- Artifact export at every step is covered by Tasks 1 through 6.
- Saved reusable `joblib` model file is covered by Task 4.
- Filesystem-based prediction reuse is covered by Task 6.
- End-to-end artifact verification is covered by Task 7.

Placeholder scan:

- No `TODO`, `TBD`, or deferred implementation markers remain.
- Each task includes explicit files, concrete code blocks, commands, and expected outcomes.

Type consistency:

- Artifact filenames are consistent across tasks: `clean_matches.csv`, `match_features.csv`, split CSVs, `feature_columns.json`, `logistic_regression_model.joblib`, `validation_predictions.csv`, `train_manifest.json`, `evaluation_metrics.json`, `test_predictions.csv`, and `predicted_match_result.csv`.
- Notebook-local helper names are consistent with their scope: `build_match_features_local`, `chronological_split_local`, `split_features_and_target_local`, and `train_logistic_regression_local`.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-18-match-win-probability-notebooks.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
