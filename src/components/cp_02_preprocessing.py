from kfp import dsl
from kfp.dsl import Dataset, Input, Output

@dsl.component(
    packages_to_install=[
        "pandas==3.0.2",
        "scikit-learn==1.8.0",
        "joblib==1.5.3",
        "pygments==2.20.0" # CVE-2026-4539
    ],
    base_image="python:3.14",
)
def feature_preprocessing(
    features_csv: Input[Dataset],
    train_csv: Output[Dataset],
    test_csv: Output[Dataset],
    preprocessing_joblib: Output[Dataset],
):
    import pandas as pd

    df = pd.read_csv(features_csv.path)
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

    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    preprocessor = ColumnTransformer([("num", numeric_pipeline, features)])

    X_train_scaled = pd.DataFrame(preprocessor.fit_transform(X_train), columns=features)  # type: ignore
    X_test_scaled = pd.DataFrame(preprocessor.transform(X_test), columns=features)  # type: ignore

    df_train_preprocessed = pd.concat([X_train_scaled, y_train], axis=1)
    df_test_preprocessed = pd.concat([X_test_scaled, y_test], axis=1)

    import joblib

    # Save datasets
    df_train_preprocessed.to_csv(train_csv.path, index=False)
    df_test_preprocessed.to_csv(test_csv.path, index=False)

    # Save preprocessor model
    joblib.dump(preprocessor, preprocessing_joblib.path)