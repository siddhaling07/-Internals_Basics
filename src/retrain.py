import pandas as pd
import joblib
import json
import os
from sklearn.ensemble import RandomForestRegressor

old = pd.read_csv("data/training_data.csv")
new = pd.read_csv("data/new_data.csv")

df = pd.concat([old, new])

X = df.drop("bug_detection_count", axis=1)
y = df["bug_detection_count"]

model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

joblib.dump(model, "models/best_model.joblib")

os.makedirs("results", exist_ok=True)

with open("results/step4_s8.json", "w") as f:
    json.dump({
        "status": "retrained",
        "total_rows": len(df)
    }, f, indent=2)

print("Retraining done!")