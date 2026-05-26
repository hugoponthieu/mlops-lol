from kfp import dsl
from kfp.dsl import Dataset, Input, Output

@dsl.component(
    packages_to_install=[
        "pandas==2.3.3",
        "mlflow==3.12.0",
        "tensorflow==2.21.0",
        "tensorboard==2.20.0",
        "matplotlib==3.10.9",
        "tf2onnx==1.17.0",
        "scikit-learn==1.8.0",
        "python-dotenv==1.2.2",
        "numpy==2.4.4",
        "onnx==1.21.0",
        "onnxruntime==1.26.0",
        "uv==0.11.12",
        "boto3==1.43.6",
        "gitpython==3.1.50", # CVE-2026-42215, 42284, 44243, 44244, GHSA-mv93-w799-cj2w
        "mako==1.3.12", # CVE-2026-44307
        "pillow==12.2.0", # CVE-2026-40192, 42311
        "aiohttp==3.13.5", # CVE-2026-22815, 34515, 34516, 43525
        "requests==2.33.1", # CVE-2026-25645
        "pip==26.1.1", # CVE-2026-6357
        "pygments==2.20.0" # CVE-2026-4539
    ],
    base_image="python:3.12",
)
def feature_train_model(
    train_csv: Input[Dataset],
    train_ds: Output[Dataset],
    val_ds: Output[Dataset],
    model_path: Output[Dataset],
):
    import pandas as pd

    df = pd.read_csv(train_csv.path)

    target = "team_1_wins"
    X = df.drop(target, axis=1)
    y = df[target]

    import numpy as np
    import tensorflow as tf
    import tf2onnx
    import onnx
    from keras import callbacks, layers, metrics, models, optimizers
    from mlflow.models import infer_signature
    from sklearn.model_selection import train_test_split

    learning_rate = 0.01
    dense_1_units = 64
    dense_2_units = 16
    dropout = 0.3
    batch_size = 128
    epoch = 100
    validation_split = 0.2
    patience = 10

    model = models.Sequential(
        [
            layers.Input(shape=(X.shape[1],)),
            layers.Dense(dense_1_units, activation="relu"),
            layers.Dropout(dropout),
            layers.BatchNormalization(),
            layers.Dense(dense_2_units, activation="relu"),
            layers.Dropout(dropout),
            layers.BatchNormalization(),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", metrics.AUC(name="auc")],
    )

    # Save KFP output paths before variable names are reused below
    train_ds_path = train_ds.path
    val_ds_path = val_ds.path
    keras_path = model_path.path + ".keras"
    onnx_path = model_path.path + ".onnx"

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=42
    )

    tf_train_ds = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
    tf_val_ds = tf.data.Dataset.from_tensor_slices((X_val.values, y_val.values))

    tf_train_ds = tf_train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    tf_val_ds = tf_val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    model.fit(
        tf_train_ds,
        validation_data=tf_val_ds,
        epochs=epoch,
        callbacks=[callbacks.EarlyStopping(patience=patience, restore_best_weights=True)],
    )

    model.save(keras_path)

    spec = (tf.TensorSpec((None, len(X.columns)), tf.float32, name="input"),)

    @tf.function(input_signature=spec)
    def model_fn(x):
        return model(x)

    tf2onnx.convert.from_function(
        model_fn,
        opset=17,
        output_path=onnx_path,
        input_signature=spec,
    )

    train_df = pd.concat([pd.DataFrame(X_train), pd.Series(y_train, name="team_1_wins")], axis=1)
    val_df = pd.concat([pd.DataFrame(X_val), pd.Series(y_val, name="team_1_wins")], axis=1)

    train_df.to_csv(train_ds_path, index=False)
    val_df.to_csv(val_ds_path, index=False)