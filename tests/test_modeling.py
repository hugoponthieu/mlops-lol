import pandas as pd

from mlops.evaluation import summarize_probabilities
from mlops.modeling import elo_probability_from_diff, train_logistic_regression


def test_train_logistic_regression_returns_probability_model():
    X_train = pd.DataFrame(
        {
            "elo_diff": [10.0, -10.0, 25.0, -20.0],
            "winrate_last_5_diff": [0.2, -0.2, 0.4, -0.3],
            "event": ["Main", "Main", "Playoffs", "Playoffs"],
        }
    )
    y_train = pd.Series([1, 0, 1, 0], dtype="Int64")

    model = train_logistic_regression(X_train, y_train)
    probs = model.predict_proba(X_train)

    assert probs.shape == (4, 2)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_elo_probability_from_diff_is_symmetric():
    probs = elo_probability_from_diff(pd.Series([0.0, 100.0, -100.0]))

    assert probs.iloc[0] == 0.5
    assert probs.iloc[1] > 0.5
    assert probs.iloc[2] < 0.5


def test_summarize_probabilities_returns_expected_metric_keys():
    y_true = pd.Series([1, 0, 1, 0])
    y_prob = pd.Series([0.8, 0.2, 0.7, 0.1])

    metrics = summarize_probabilities(y_true, y_prob)

    assert set(metrics) == {"log_loss", "brier_score", "roc_auc", "accuracy_50"}
