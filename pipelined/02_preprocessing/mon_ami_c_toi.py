from kfp.dsl import component, Input, Output, Dataset


@component(
    base_image="python:3.14",
    packages_to_install=["pandas==3.01", "sklearn", "joblib", "os"],
)
def preprocessing(
    input_results_dataset: Input[Dataset],
    train_data_set: Output[Dataset],
    test_data_set: Output[Dataset],
    preprocessing_joblib: Output[Dataset],
):
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    import joblib
    import os

    df = pd.read_csv(input_results_dataset.path)

    df["date"] = pd.to_datetime(df["date"])
    split_date = df["date"].quantile(0.9)
    df_train = df[df["date"] <= split_date].copy()
    df_test = df[df["date"] > split_date].copy().reset_index()

    features = [
        "elo_diff",
        "winrate_10_diff",
        "winrate_30_diff",
        "experience_diff",
        "rank_diff",
        "h2h_winrate",
    ]
    target = "team_1_wins"

    X_train = df_train[features]
    y_train = df_train[target]
    X_test = df_test[features]
    y_test = df_test[target]

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    preprocessor = ColumnTransformer([("num", numeric_pipeline, features)])

    X_train_scaled = pd.DataFrame(preprocessor.fit_transform(X_train), columns=features)  # type: ignore
    X_test_scaled = pd.DataFrame(preprocessor.transform(X_test), columns=features)  # type: ignore

    df_train_preprocessed = pd.concat([X_train_scaled, y_train], axis=1)
    df_test_preprocessed = pd.concat([X_test_scaled, y_test], axis=1)

    # Save datasets

    df_train_preprocessed.to_csv(train_data_set.path, index=False)
    df_test_preprocessed.to_csv(test_data_set.path, index=False)

    joblib.dump(preprocessor, preprocessing_joblib.path)
