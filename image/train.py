import os

import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.ensemble import GradientBoostingRegressor

dataset = load_dataset("codesignal/tsla-historic-prices", split="train")
df = pd.DataFrame(dataset)

df["SMA_20"] = df["Close"].rolling(window=20).mean()

df = df.dropna()

X = df[["SMA_20"]]
y = df["Close"]

model = GradientBoostingRegressor()
model.fit(X, y)

X_future = pd.DataFrame({"SMA_20": [df["Close"].iloc[-20:].mean()]})

predicted_close_today = model.predict(X_future)

# Save model to S3
model_output_dir = "/opt/ml/model"
model_file_path = os.path.join(model_output_dir, "model.joblib")
os.makedirs(model_output_dir, exist_ok=True)
joblib.dump(model, model_file_path)
