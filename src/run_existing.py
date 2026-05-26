"""Trigger a run on an already-uploaded pipeline (no recompile).

Usage:
    PIPELINE_ID=<id> uv run python src/run_existing.py
    PIPELINE_ID=<id> ARG_learning_rate=0.005 ARG_epochs=20 \
        RUN_NAME=lr-low EXPERIMENT_NAME=hpo uv run python src/run_existing.py
"""

from main import run_existing

if __name__ == "__main__":
    run_existing()
