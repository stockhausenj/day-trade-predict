import os

import joblib
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def fetch_data(symbol="IBM", apikey="demo"):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={apikey}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
    else:
        raise Exception("Failed to fetch stock data...")

    time_series_data = data["Time Series (Daily)"]
    data_list = []
    for date, metrics in time_series_data.items():
        metrics["date"] = date
        data_list.append(metrics)
    data_list.reverse()
    return data_list


def train(df, model, scaler):
    features = df[
        [
            "Y_Close",
            "Y_SMA_20",
            "Y_EMA_10",
        ]
    ].values
    target = df["close"].values

    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, target, test_size=0.25, random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    # TODO: Stuck at around 4.5 for now. Get to below 1.0
    print("Mean Squared Error:", mse)


def feature_dtypes(df):
    float_columns = ["open", "close", "high", "low", "Y_EMA_5", "Y_SMA_20", "Y_Close"]
    for float_column in float_columns:
        if float_column in df:
            df[float_column] = df[float_column].astype("float64")

    int_columns = ["volume"]
    for int_column in int_columns:
        if int_column in df:
            df[int_column] = df[int_column].astype("int64")


df = pd.DataFrame(fetch_data())
df.columns = df.columns.str.replace(r"^\d+\.\s*", "", regex=True)

feature_dtypes(df)

df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

df["Y_Close"] = df["close"].shift(1)

"""
The EMA reacts faster to price changes, capturing recent trends, while the 20-day SMA will offer a smoother 
  perspective on medium-term trends. Together, these can signal trend shifts without being overly sensitive 
  to short-term volatility.
"""
df["SMA_20"] = df["close"].rolling(window=10).mean()
df["Y_SMA_20"] = df["SMA_20"].shift(1)
df["EMA_10"] = df["close"].ewm(span=10, adjust=False).mean()
df["Y_EMA_10"] = df["EMA_10"].shift(1)

# Delete columns not used. Should be features + target.
df = df[["Y_Close", "Y_SMA_20", "Y_EMA_10", "close"]].dropna()

model = GradientBoostingRegressor(
    n_estimators=500, max_depth=4, learning_rate=0.05, random_state=42
)

scaler = StandardScaler()

train(df, model, scaler)

predictions = []
historical_closes = df["close"].iloc[-20:].tolist()

# Predict 14 days of closing values
for _ in range(14):
    # Create next day's features
    sma_20 = sum(historical_closes[-20:]) / 20
    ema_10 = pd.Series(historical_closes[-10]).ewm(span=10, adjust=False).mean().iloc[0]
    y_close = historical_closes[-1]
    features_future = pd.DataFrame(
        {
            "Y_EMA_10": [ema_10],
            "Y_SMA_20": [sma_20],
            "Y_Close": [y_close],
        }
    )
    features_future_scaled = scaler.transform(features_future.values)

    predicted_close = model.predict(features_future_scaled)
    predictions.append(predicted_close)

    features_future["close"] = predicted_close
    df = pd.concat([df, features_future], ignore_index=True)
    train(df, model, scaler)
    historical_closes.append(predicted_close)

prediction_output_dir = "/opt/ml/output/data"
os.makedirs(prediction_output_dir, exist_ok=True)
prediction_file_path = os.path.join(prediction_output_dir, "predictions.csv")
predictions.to_csv(prediction_file_path, index=False)

# Save model to S3. SageMaker requires model to be in /opt/ml/model/
model_output_dir = "/opt/ml/model"
model_file_path = os.path.join(model_output_dir, "model.joblib")
os.makedirs(model_output_dir, exist_ok=True)
joblib.dump(model, model_file_path)
