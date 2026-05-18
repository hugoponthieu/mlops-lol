# Match Win Probability Notebook Design

## Goal

Create a series of Jupyter notebooks to estimate pre-match win probabilities from the historical match dataset in `data/matchs_stats.csv`.

The notebooks should be easy to use, readable, and organized by training stage. Each step must export filesystem artifacts that the next step can reuse, and the trained model must be saved to disk rather than existing only in memory.

## Confirmed Scope

- Prediction target: pre-match win probability
- Main outcome: estimate `P(blue_team wins)`
- Modeling style: classical baseline first
- Workflow style: notebook-first
- Code organization: notebook-self-contained
- Persistence style: explicit exported artifacts between every step
- Feature policy: strict pre-match only
- Entity level: team-only features for the first version

## Dataset Notes

The available dataset is `data/matchs_stats.csv`. Although the initial request referenced `data/players_stats.csv`, the current workspace contains only the match-level file.

Important columns observed in the dataset:

- Match context: `season`, `date`, `event`, `patch`
- Teams: `blue_team`, `red_team`
- Label: `winner`
- Draft information: bans and picks for both teams
- Player names: role-specific player columns for both teams

The first version will use only information known before match start. That means current-match draft columns and player identity columns are excluded from the modeling feature set.

## Notebook Series

### 01_data_pull_and_audit.ipynb

Purpose:

- Load `data/matchs_stats.csv`
- Inspect schema and missingness
- Parse and validate dates
- Normalize obvious formatting issues
- Sort data chronologically
- Define the target column `blue_team_win`
- Document which columns are allowed and disallowed under the strict pre-match rule

Outputs:

- `artifacts/clean_matches.csv`
- Data quality observations
- A documented leakage policy for later notebooks

### 02_feature_engineering.ipynb

Purpose:

- Build historical team-level features using only prior matches
- Reuse the Elo idea already present in `examples/feature.py`, but align it to the real schema
- Ensure each match row is transformed without using future information

Recommended first-pass features:

- `elo_diff`: blue Elo minus red Elo
- `winrate_last_5_diff`
- `winrate_last_10_diff`
- `winrate_last_20_diff`
- `matches_played_diff`
- `days_since_last_match_diff`
- `head_to_head_winrate_diff`
- `blue_side_team_winrate`
- `red_side_team_winrate`
- Optional context descriptors known before the match: `patch`, `event`, `season`

Outputs:

- `artifacts/match_features.csv`
- Leakage-safe chronological feature generation logic

### 03_preprocessing.ipynb

Purpose:

- Perform chronological train, validation, and test splits
- Handle missing values introduced by sparse early history
- Encode categorical pre-match context features if retained
- Separate modeling inputs from labels

Rules:

- No random split
- No shuffling across time
- Validation and test periods must happen strictly after training data

Outputs:

- `artifacts/train_features.csv`
- `artifacts/valid_features.csv`
- `artifacts/test_features.csv`
- `artifacts/train_labels.csv`
- `artifacts/valid_labels.csv`
- `artifacts/test_labels.csv`
- `artifacts/feature_columns.json`

### 04_train_probability_model.ipynb

Purpose:

- Train a baseline probabilistic model
- Compare a simple Elo-only benchmark against a richer engineered-feature model

Recommended model order:

1. Elo-only probability baseline
2. Logistic regression baseline
3. Optional second classical model if needed, such as gradient boosting

Training goals:

- Produce calibrated or calibratable win probabilities
- Keep the first model interpretable
- Preserve a workflow that is easy to rerun in Jupyter

Outputs:

- `artifacts/logistic_regression_model.joblib`
- `artifacts/validation_predictions.csv`
- `artifacts/train_manifest.json`

### 05_evaluation_and_calibration.ipynb

Purpose:

- Evaluate the quality of predicted probabilities, not just ranking performance
- Measure calibration and reliability

Recommended metrics:

- Log loss
- Brier score
- ROC-AUC
- Accuracy at a chosen threshold, only as a secondary metric
- Calibration curve / reliability diagram

Recommended calibration step:

- If logistic regression is not well calibrated on the validation split, test a calibration layer such as Platt scaling or isotonic regression using validation data only

Outputs:

- `artifacts/evaluation_metrics.json`
- `artifacts/test_predictions.csv`
- Optional plot files under `artifacts/plots/`
- Clear recommendation on which model version to carry forward

### 06_predict_match.ipynb

Purpose:

- Load the persisted cleaned history and persisted trained model from disk
- Build one future match row using the same pre-match feature rules
- Predict the result without retraining inside the same notebook session

Inputs:

- `artifacts/clean_matches.csv`
- `artifacts/logistic_regression_model.joblib`
- `artifacts/feature_columns.json`

Outputs:

- `artifacts/predicted_match_result.csv`

## Data Flow

1. Load raw match data
2. Clean and sort chronologically, then save `artifacts/clean_matches.csv`
3. For each match, compute features from prior history only, then save `artifacts/match_features.csv`
4. Build time-based splits and save training, validation, and test artifacts
5. Train the baseline probability model and save a reusable `joblib` model file
6. Evaluate and calibrate probabilities, then save metrics and predictions
7. Load the persisted model and predict a future matchup, then save the result

This keeps the entire workflow consistent with the pre-match prediction constraint.

## Artifact Strategy

All notebook outputs should be explicit files under `artifacts/` at the repository root.

Formats:

- CSV for all tabular datasets exchanged between notebooks
- `joblib` for the trained scikit-learn model
- JSON for small metadata artifacts such as feature column lists and metric summaries

This keeps each stage inspectable and restartable. A user should be able to begin from any notebook as long as the prior exported artifacts already exist on disk.

## Leakage Rules

The design depends on strict leakage control.

Allowed for the current match row:

- `blue_team`
- `red_team`
- `date`
- `event`
- `patch`
- Any feature derived only from matches dated earlier than the current match

Not allowed for the current match row:

- `winner`
- Current-match picks and bans
- Current-match player assignments
- Any aggregate accidentally computed using the full dataset instead of prior history only

## Error Handling And Edge Cases

Expected issues and handling:

- Missing `patch` values: keep as missing or map to `"unknown"` consistently
- Sparse early history: use default values and smoothing for Elo, rolling win rates, and head-to-head features
- Team naming inconsistencies: inspect and normalize only if concrete duplicates are found
- Same-day matches: preserve stable chronological ordering and document the ordering assumption if timestamps are unavailable

## Testing Strategy

The notebook series should include lightweight checks at each stage:

- Data audit checks for expected columns and parsable dates
- Artifact checks that confirm each expected file is written where documented
- Feature checks that confirm no null explosions or impossible values
- Chronology checks that ensure each feature uses prior history only
- Split checks that confirm train dates precede validation and test dates
- Evaluation checks that compare Elo-only and model-based probabilities on the same holdout sets

## Recommended Structure In Practice

The notebooks should be self-sufficient. They may define helper functions in top cells, but they should not depend on local Python package modules to run the core workflow.

That means:

- One main purpose per notebook
- Shared assumptions repeated briefly where needed
- Stable helper functions grouped near the top of a notebook
- Read the prior stage's exported artifacts from `artifacts/`
- Write the current stage's outputs back to `artifacts/`
- Keep any repeated logic small and visible inside the notebook rather than hiding it in required imports

## Non-Goals For Version 1

To keep the scope controlled, the first version should not include:

- TensorFlow models
- Player-level historical features
- Current-draft-aware prediction
- Multi-stage modeling before and after draft
- Full packaging into a production training pipeline

## Success Criteria

This design is successful if it produces:

- A clean notebook sequence from raw data to evaluated probabilities
- Exported files at every stage that can be reused by later notebooks
- A saved model file that can be loaded again outside the original notebook RAM state
- A leakage-safe feature table built chronologically
- A baseline model that outputs usable pre-match win probabilities
- Evaluation artifacts that make model quality easy to interpret

## Implementation Direction After Approval

After the spec is approved, the next step should be a written implementation plan that breaks the notebook work into concrete tasks, notebook contents, and verification steps.
