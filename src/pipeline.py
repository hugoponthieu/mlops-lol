from kfp import dsl, kubernetes

from components.cp_00_pull_data import pull_data
from components.cp_01_feature_engineering import feature_engineering
from components.cp_02_preprocessing import feature_preprocessing
from components.cp_03_train_model import feature_train_model

@dsl.pipeline
def cs_go_match_predictor_train():
    pull_data_task = pull_data()

    kubernetes.use_secret_as_env(
        pull_data_task,
        secret_name="dvc-credentials",
        secret_key_to_env={"DVC_GIT_REPO": "DVC_GIT_REPO", "DVC_S3_ACCESS_KEY": "DVC_S3_ACCESS_KEY", "DVC_S3_SECRET_KEY": "DVC_S3_SECRET_KEY"},
    )

    kubernetes.use_secret_as_volume(
        pull_data_task,
        secret_name="dvc-ssh-key",
        mount_path="/root/.ssh",
    )

    feature_engineering_task = feature_engineering(matches_csv=pull_data_task.outputs["matches_csv"])
    processing_task = feature_preprocessing(features_csv=feature_engineering_task.outputs["features_csv"])
    _train_model_task = feature_train_model(train_csv=processing_task.outputs["train_csv"])