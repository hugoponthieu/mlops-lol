from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output

@dsl.component(
    packages_to_install=[
        "kubeflow==0.4.0",
        "boto3==1.43.6",
        "pandas==2.3.3",
        "requests==2.33.1", # CVE-2026-25645
        "urllib3==2.7.0",
        "pygments==2.20.0" # CVE-2026-4539
    ],
    base_image="python:3.12",
)
def feature_train_model(
    train_csv: Input[Dataset],
    test_csv: Input[Dataset],
    model: Output[Model],
    s3_bucket: str = "mlops-bif",
    s3_endpoint: str = "https://b09595e7413d52541d55ebc0bf445b9c.r2.cloudflarestorage.com",
    s3_region: str = "auto",
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
    import json
    import os
    import time
    import uuid

    import boto3
    from kubeflow.trainer import (
        CustomTrainer,
        Initializer,
        S3DatasetInitializer,
        TrainerClient,
    )

    access_key = os.environ["DVC_S3_ACCESS_KEY"]
    secret_key = os.environ["DVC_S3_SECRET_KEY"]

    run_id = uuid.uuid4().hex[:10]
    data_prefix = f"{s3_prefix}/{run_id}/data"
    model_prefix = f"{s3_prefix}/{run_id}/model"

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=s3_region,
    )
    s3.upload_file(train_csv.path, s3_bucket, f"{data_prefix}/train.csv")
    s3.upload_file(test_csv.path, s3_bucket, f"{data_prefix}/test.csv")
    print(f"Uploaded preprocessed CSVs to s3://{s3_bucket}/{data_prefix}/")

    def train_fn(
        learning_rate: float,
        dense_1_units: int,
        dense_2_units: int,
        dropout: float,
        batch_size: int,
        epochs: int,
        patience: int,
        validation_split: float,
        s3_bucket: str,
        s3_endpoint: str,
        s3_region: str,
        model_prefix: str,
        access_key: str,
        secret_key: str,
    ):
        import json
        import os

        import boto3
        import pandas as pd
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset, random_split

        target = "team_1_wins"
        df = pd.read_csv("/workspace/dataset/train.csv")
        feature_columns = [c for c in df.columns if c != target]
        n_features = len(feature_columns)

        X = torch.tensor(df[feature_columns].values, dtype=torch.float32)
        y = torch.tensor(df[target].values, dtype=torch.float32).unsqueeze(1)
        full_ds = TensorDataset(X, y)

        val_size = max(1, int(len(full_ds) * validation_split))
        train_size = len(full_ds) - val_size
        train_ds, val_ds = random_split(
            full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size)

        net = nn.Sequential(
            nn.Linear(n_features, dense_1_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(dense_1_units),
            nn.Linear(dense_1_units, dense_2_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(dense_2_units),
            nn.Linear(dense_2_units, 1),
        )
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        best_state = None
        epochs_since_improve = 0
        final_metrics = {}

        for epoch in range(epochs):
            net.train()
            train_loss_sum, train_n = 0.0, 0
            for xb, yb in train_dl:
                optimizer.zero_grad()
                logits = net(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() * xb.size(0)
                train_n += xb.size(0)
            train_loss = train_loss_sum / max(train_n, 1)

            net.eval()
            val_loss_sum, val_correct, val_n = 0.0, 0, 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    logits = net(xb)
                    val_loss_sum += loss_fn(logits, yb).item() * xb.size(0)
                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    val_correct += (preds == yb).sum().item()
                    val_n += xb.size(0)
            val_loss = val_loss_sum / max(val_n, 1)
            val_acc = val_correct / max(val_n, 1)
            print(
                f"epoch {epoch+1}/{epochs} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
            final_metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= patience:
                    print(f"Early stop at epoch {epoch+1}")
                    break

        if best_state is not None:
            net.load_state_dict(best_state)

        out_dir = "/tmp/model_out"
        os.makedirs(out_dir, exist_ok=True)
        torch.save(net.state_dict(), os.path.join(out_dir, "model.pt"))

        net.eval()
        dummy = torch.randn(1, n_features)
        torch.onnx.export(
            net,
            dummy,
            os.path.join(out_dir, "model.onnx"),
            input_names=["input"],
            output_names=["logits"],
            opset_version=17,
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        )

        metadata = {
            "framework": "pytorch",
            "framework_version": torch.__version__,
            "task": "binary_classification",
            "input_features": feature_columns,
            "input_shape": [None, n_features],
            "pt_file": "model.pt",
            "onnx_file": "model.onnx",
            "onnx_opset": 17,
            "hyperparameters": {
                "learning_rate": learning_rate,
                "dense_1_units": dense_1_units,
                "dense_2_units": dense_2_units,
                "dropout": dropout,
                "batch_size": batch_size,
                "epochs": epochs,
                "patience": patience,
                "validation_split": validation_split,
            },
            "final_metrics": final_metrics,
        }
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        s3c = boto3.client(
            "s3",
            endpoint_url=s3_endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=s3_region,
        )
        for fname in ("model.pt", "model.onnx", "metadata.json"):
            s3c.upload_file(os.path.join(out_dir, fname), s3_bucket, f"{model_prefix}/{fname}")
        print(f"Uploaded model artifacts to s3://{s3_bucket}/{model_prefix}/")

    client = TrainerClient()
    job_name = client.train(
        runtime=client.get_runtime(runtime_name),
        trainer=CustomTrainer(
            func=train_fn,
            func_args={
                "learning_rate": learning_rate,
                "dense_1_units": dense_1_units,
                "dense_2_units": dense_2_units,
                "dropout": dropout,
                "batch_size": batch_size,
                "epochs": epochs,
                "patience": patience,
                "validation_split": validation_split,
                "s3_bucket": s3_bucket,
                "s3_endpoint": s3_endpoint,
                "s3_region": s3_region,
                "model_prefix": model_prefix,
                "access_key": access_key,
                "secret_key": secret_key,
            },
            packages_to_install=[
                "torch",
                "pandas",
                "boto3",
                "onnx",
            ],
            num_nodes=num_nodes,
        ),
        initializer=Initializer(
            dataset=S3DatasetInitializer(
                storage_uri=f"s3://{s3_bucket}/{data_prefix}",
                endpoint=s3_endpoint,
                access_key_id=access_key,
                secret_access_key=secret_key,
                region=s3_region,
            )
        ),
    )
    print(f"Submitted TrainJob: {job_name}")

    terminal_success = {"succeeded", "complete"}
    terminal_failure = {"failed", "error"}
    while True:
        info = client.get_job(job_name)
        status_obj = getattr(info, "status", info)
        status_str = str(status_obj).lower()
        print(f"TrainJob status: {status_obj}")
        if any(s in status_str for s in terminal_success):
            break
        if any(s in status_str for s in terminal_failure):
            try:
                for line in client.get_job_logs(job_name):
                    print(line)
            except Exception as e:
                print(f"(could not stream logs: {e})")
            raise RuntimeError(f"TrainJob {job_name} failed with status {status_obj}")
        time.sleep(15)

    try:
        for line in client.get_job_logs(job_name):
            print(line)
    except Exception as e:
        print(f"(log streaming after completion failed: {e})")

    os.makedirs(model.path, exist_ok=True)
    for fname in ("model.pt", "model.onnx", "metadata.json"):
        s3.download_file(s3_bucket, f"{model_prefix}/{fname}", os.path.join(model.path, fname))

    with open(os.path.join(model.path, "metadata.json")) as f:
        md = json.load(f)
    for k, v in md.items():
        model.metadata[k] = v
    model.metadata["trainjob_name"] = job_name
    model.metadata["s3_model_uri"] = f"s3://{s3_bucket}/{model_prefix}"
