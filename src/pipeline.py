from kfp import dsl, kubernetes

from components.cp_00_pull_data import pull_data
from components.cp_01_feature_engineering import feature_engineering
from components.cp_02_preprocessing import feature_preprocessing
from components.cp_03_train_model import feature_train_model

@dsl.pipeline
def cs_go_match_predictor_train(
    s3_bucket: str = "mlops-bif",
    s3_endpoint: str = "https://b09595e7413d52541d55ebc0bf445b9c.r2.cloudflarestorage.com",
    s3_region: str = "auto",
    s3_dataset_key: str = "data/raw/dataset.csv",
    s3_prefix: str = "kfp-runs",
    runtime_name: str = "torch-distributed",
    num_nodes: int = 1,
    learning_rate: float = 0.001,
    dense_1_units: int = 64,
    dense_2_units: int = 16,
    dropout: float = 0.3,
    batch_size: int = 128,
    epochs: int = 100,
    patience: int = 10,
    validation_split: float = 0.2,
):
    pull_data_task = pull_data(
        s3_bucket=s3_bucket,
        s3_key=s3_dataset_key,
        s3_endpoint=s3_endpoint,
        s3_region=s3_region,
    )
    kubernetes.use_secret_as_env(
        pull_data_task,
        secret_name="dvc-credentials",
        secret_key_to_env={
            "DVC_S3_ACCESS_KEY": "DVC_S3_ACCESS_KEY",
            "DVC_S3_SECRET_KEY": "DVC_S3_SECRET_KEY",
        },
    )

    feature_engineering_task = feature_engineering(
        matches_csv=pull_data_task.outputs["matches_csv"],
    )
    processing_task = feature_preprocessing(
        features_csv=feature_engineering_task.outputs["features_csv"],
    )
    train_task = feature_train_model(
        train_csv=processing_task.outputs["train_csv"],
        test_csv=processing_task.outputs["test_csv"],
        s3_bucket=s3_bucket,
        s3_endpoint=s3_endpoint,
        s3_region=s3_region,
        s3_prefix=s3_prefix,
        runtime_name=runtime_name,
        num_nodes=num_nodes,
        learning_rate=learning_rate,
        dense_1_units=dense_1_units,
        dense_2_units=dense_2_units,
        dropout=dropout,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        validation_split=validation_split,
    )
    kubernetes.use_secret_as_env(
        train_task,
        secret_name="dvc-credentials",
        secret_key_to_env={
            "DVC_S3_ACCESS_KEY": "DVC_S3_ACCESS_KEY",
            "DVC_S3_SECRET_KEY": "DVC_S3_SECRET_KEY",
        },
    )
