import os

import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.ensemble import GradientBoostingRegressor

# Load dataset
dataset = load_dataset("codesignal/tsla-historic-prices", split="train")
df = pd.DataFrame(dataset)

# Preprocessing
df["SMA_20"] = df["Close"].rolling(window=20).mean()
df = df.dropna()

# Define features and target
features = df[["SMA_20"]]
target = df["Close"]

# Train
model = GradientBoostingRegressor()
model.fit(features, target)

# Predict and save prediction
features_future = pd.DataFrame({"SMA_20": [df["Close"].iloc[-20:].mean()]})
predicted_close_today = model.predict(features_future)

prediction_df = features_future.copy()
prediction_df["Predicted_Close"] = predicted_close_today

prediction_output_dir = "/opt/ml/output/data"
os.makedirs(prediction_output_dir, exist_ok=True)
prediction_file_path = os.path.join(prediction_output_dir, "predictions.csv")
prediction_df.to_csv(prediction_file_path, index=False)

# Save model to S3. SageMaker requires model to be in /opt/ml/model/
model_output_dir = "/opt/ml/model"
model_file_path = os.path.join(model_output_dir, "model.joblib")
os.makedirs(model_output_dir, exist_ok=True)
joblib.dump(model, model_file_path)
