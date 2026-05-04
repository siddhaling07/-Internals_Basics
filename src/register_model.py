import mlflow
import mlflow.sklearn
import joblib
import json
import os

MODEL_PATH = "models/best_model.joblib"
RESULTS_PATH = "results/step3_s6.json"
MODEL_NAME = "bug_model"

# Load model
model = joblib.load(MODEL_PATH)

# Register model in MLflow
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name=MODEL_NAME
    )

    run_id = run.info.run_id

print("Model registered!")

# Save output JSON (for marks)
os.makedirs("results", exist_ok=True)

output = {
    "model_name": MODEL_NAME,
    "run_id": run_id,
    "stage": "None",
    "status": "registered"
}

with open(RESULTS_PATH, "w") as f:
    json.dump(output, f, indent=2)

print("Saved results/step3_s6.json")