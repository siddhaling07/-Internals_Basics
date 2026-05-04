import json, os, joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

FEATURES = ["pr_lines_changed", "file_count", "author_experience_months", "is_refactor"]
TARGET   = "bug_detection_count"

DATA_PATH = "data/training_data.csv"
MODEL_DIR = "models"
RESULTS_DIR = "results"
EXPERIMENT_NAME = "reviewbot_bug_detection"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # ✅ FIX
    r2 = r2_score(y_true, y_pred)
    return {"mae": round(mae,4), "rmse": round(rmse,4), "r2": round(r2,4)}

def make_models():
    return {
        "poisson": Pipeline([
            ("scaler", StandardScaler()),
            ("model", PoissonRegressor(max_iter=1000))
        ]),
        "rf": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        "gboost": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
    }

def run():
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_data()

    results = {}
    best_model = None
    best_mae = float("inf")

    for name, model in make_models().items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            preds = np.clip(preds, 0, None)

            metrics = evaluate(y_test, preds)

            mlflow.log_params({"model": name})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

            results[name] = metrics

            print(name, metrics)

            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                best_model = model

    joblib.dump(best_model, "models/best_model.joblib")

    with open("results/step1_s1.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Training complete!")

if __name__ == "__main__":
    run()