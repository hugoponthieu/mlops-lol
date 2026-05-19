from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


def summarize_probabilities(y_true: pd.Series, y_prob: pd.Series) -> Dict[str, float]:
    labels = y_true.astype(int)
    probs = y_prob.astype(float)
    return {
        "log_loss": float(log_loss(labels, probs)),
        "brier_score": float(brier_score_loss(labels, probs)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "accuracy_50": float(accuracy_score(labels, (probs >= 0.5).astype(int))),
    }
