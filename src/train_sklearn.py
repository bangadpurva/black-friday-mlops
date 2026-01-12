# src/train_sklearn.py

import os
import mlflow
import mlflow.sklearn
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import expr

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor  # strong + compact

from mlflow.models.signature import infer_signature
from mlflow import register_model

spark = SparkSession.builder.getOrCreate()

LABEL_COL = "Purchase"

CATEGORICAL_COLS = ["Gender", "Age", "City_Category", "Stay_In_Current_City_Years"]
NUMERIC_COLS = [
    "Occupation",
    "Marital_Status",
    "Product_Category_1",
    "Product_Category_2",
    "Product_Category_3",
]

REQUIRED_COLS = CATEGORICAL_COLS + NUMERIC_COLS + [LABEL_COL]


def load_and_clean(table_name: str, sample_n: int | None = None) -> pd.DataFrame:
    df = spark.table(table_name)

    # Safe numeric casting: converts malformed values like "null" -> NULL
    for c in NUMERIC_COLS + [LABEL_COL]:
        df = df.withColumn(c, expr(f"try_cast({c} as double)"))

    df = df.select(*REQUIRED_COLS)

    # Optional sampling to avoid driver OOM on very large tables
    if sample_n is not None and sample_n > 0:
        df = df.limit(sample_n)

    return df.toPandas()


def build_pipeline() -> Pipeline:
    cat_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    num_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_tf, CATEGORICAL_COLS),
            ("num", num_tf, NUMERIC_COLS),
        ]
    )

    model = HistGradientBoostingRegressor(
        random_state=42,
        max_depth=None,        # can tune later
        learning_rate=0.08,
        max_iter=int(os.getenv("MAX_ITER", "300")),
    )

    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def main():
    train_table = os.getenv("TRAIN_TABLE", "main.default.train")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "/Shared/black-friday-mlops")

    # If your table is large, set SAMPLE_N (e.g., 200000). 0/empty means full data.
    sample_n_env = os.getenv("SAMPLE_N", "0").strip()
    sample_n = int(sample_n_env) if sample_n_env.isdigit() else 0

    mlflow.set_experiment(experiment_name)

    pdf = load_and_clean(train_table, sample_n=sample_n if sample_n > 0 else None)

    # Split
    X = pdf[CATEGORICAL_COLS + NUMERIC_COLS]
    y = pdf[LABEL_COL].astype(float)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = build_pipeline()

    with mlflow.start_run():
        mlflow.log_param("train_table", train_table)
        mlflow.log_param("sample_n", sample_n if sample_n > 0 else "full")
        mlflow.log_param("model_type", "HistGradientBoostingRegressor")
        mlflow.log_param("max_iter", int(os.getenv("MAX_ITER", "300")))

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)

        mse = mean_squared_error(y_val, preds)
        rmse = mse ** 0.5
        r2 = r2_score(y_val, preds)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Add signature + input example (best practice)
        input_example = X_train.head(5)
        pred_example = pipe.predict(input_example)

        signature = infer_signature(X_train, pipe.predict(X_train.head(50)))

        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            input_example=X_train.head(5),
            signature=signature
        )

        registered_name = "main.default.black_friday_purchase_model"  # change catalog/schema if needed
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"

        register_model(model_uri, registered_name)
        print("Registered:", registered_name)

        print(f"Done. MSE={mse:.4f}, R2={r2:.4f}")


if __name__ == "__main__":
    main()
