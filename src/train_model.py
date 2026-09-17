"""Train a linear regression on the admission data and save it to the BentoML Model Store.

Usage (from the project root, after prepare_data.py):
    python src/train_model.py
"""

from pathlib import Path

import bentoml
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_NAME = "admission_lr"


def main() -> None:
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    # ravel() flattens the (n, 1) column into the 1-D array scikit-learn expects.
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").values.ravel()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").values.ravel()
    print(f"train: {X_train.shape} | test: {X_test.shape}")

    # Linear regression: the target is continuous and the features are numeric scores.
    # No scaling needed - it does not change the predictions of a linear model.
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate on the test set.
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("\nperformance on the test set")
    print(f"  R2   = {r2:.4f}")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")

    # Save the model in the BentoML Model Store so the service can load it by tag.
    # The metadata keeps the metrics and the feature order with the model.
    saved = bentoml.sklearn.save_model(
        MODEL_NAME,
        model,
        metadata={
            "r2": float(r2),
            "mae": float(mae),
            "rmse": float(rmse),
            "features": list(X_train.columns),
        },
    )
    print(f"\nmodel saved in the BentoML Model Store: {saved.tag}")
    print("verify with: bentoml models list")


if __name__ == "__main__":
    main()
