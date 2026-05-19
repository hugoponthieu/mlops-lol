from kfp.dsl import component, Input, Output, Dataset

@component(
    base_image="python:3.14",
    packages_to_install=["pandas==3.01", "numpy", "tensorflow", "tf2onnx", "onnx", "keras", "mlflow.models", "sklearn.model_selection"],
)
def train_model(input_results_dataset: Input[Dataset], trained_results_dataset: Output[Dataset]):
    import pandas as pd

    df = pd.read_csv(f"{input_results_dataset.path}/preprocessed/train.csv")

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

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=42
    )

    train_ds = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val.values, y_val.values))

    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epoch,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_auc",
                patience=patience,
                restore_best_weights=True,
            )
        ],
    )

    input_example = X_train.iloc[:5].astype(np.float32).to_numpy()
    predictions = model.predict(input_example)

    signature = infer_signature(input_example, predictions)

    model.save(f"{MODELS_DIR}/model.keras")

    spec = (tf.TensorSpec((None, len(X.columns)), tf.float32, name="input"),)

    @tf.function(input_signature=spec)
    def model_fn(x):
        return model(x)

    onnx_path = f"{MODELS_DIR}/model/cs-go-match-predictor/1/model.onnx"

    tf2onnx.convert.from_function(
        model_fn,
        opset=17,
        output_path=onnx_path,
        input_signature=spec,
    )

    onnx_model = onnx.load(onnx_path)

    X_train_df = pd.DataFrame(X_train)
    y_train_df = pd.Series(y_train, name="team_1_wins")

    train_ds = pd.concat([X_train_df, y_train_df], axis=1)

    X_val_df = pd.DataFrame(X_val)
    y_val_df = pd.Series(y_val, name="team_1_wins")

    val_ds = pd.concat([X_val_df, y_val_df], axis=1)

    train_ds.to_csv(f"{trained_results_dataset.path}/train/train.csv", index=False)
    val_ds.to_csv(f"{trained_results_dataset.path}/train/val.csv", index=False)
