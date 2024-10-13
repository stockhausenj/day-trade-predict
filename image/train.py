"""
Goals
- Predict 14 future close values for a stock.

Design
- Scale features to prevent feature dominance.
- Re-train model for each prediction.
- Measure mean squared error after each training.
- Measure future importance after each training.
- Use GridSearchCV to find optimal hyperparameters.

Feature strategy
- The EMA reacts faster to price changes, capturing recent trends, while the 20-day SMA will offer a smoother 
  perspective on medium-term trends. Together, these can signal trend shifts without being overly sensitive 
  to short-term volatility.
"""

import json
import os

import joblib
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


def feature_importance(model, feature_names):
    feature_importance = model.best_estimator_.feature_importances_

    feature_importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importance}
    )

    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    )

    print("Feature importance:\n", feature_importance_df)


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


def feature_dtypes(df):
    updated_df = df.copy()
    float_columns = ["open", "close", "high", "low", "Y_EMA_5", "Y_SMA_20", "Y_Close"]
    for float_column in float_columns:
        if float_column in updated_df:
            updated_df[float_column] = updated_df[float_column].astype("float64")

    int_columns = ["volume"]
    for int_column in int_columns:
        if int_column in updated_df:
            updated_df[int_column] = updated_df[int_column].astype("int64")
    return updated_df


def preprocess(df):
    processed_df = df.copy()
    processed_df.columns = processed_df.columns.str.replace(
        r"^\d+\.\s*", "", regex=True
    )

    processed_df = feature_dtypes(processed_df)

    processed_df["date"] = pd.to_datetime(df["date"])
    processed_df.set_index("date", inplace=True)

    processed_df["Y_Close"] = processed_df["close"].shift(1)
    processed_df["SMA_20"] = processed_df["close"].rolling(window=10).mean()
    processed_df["Y_SMA_20"] = processed_df["SMA_20"].shift(1)
    processed_df["EMA_10"] = processed_df["close"].ewm(span=10, adjust=False).mean()
    processed_df["Y_EMA_10"] = processed_df["EMA_10"].shift(1)

    processed_df = processed_df[["Y_Close", "Y_SMA_20", "Y_EMA_10", "close"]].dropna()

    return processed_df


def mse(model, X_test, y_test):
    mse_predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, mse_predictions)
    # TODO: Stuck at around 4.5 for now. Get to below 1.0
    print("Mean Squared Error:", mse)


def train(df, scaler):
    feature_names = ["Y_Close", "Y_SMA_20", "Y_EMA_10"]
    """
    n_estimators        = Number of trees to be used in the model. More can increase performance but could lead 
                          to overfitting.
    max_depth           = Max depth of trees. Deeper trees can capture more complex patters but may lead to 
                          overfitting.
    learning_rate       = Contribution of each tree in final prediction. Smaller rate means the model learns 
                          more slowly.
    validation_fraction = Fraction of data to be used as a validation set for early stopping.
    n_iter_no_change    = Number of iterations with no improvement to wait before stopping.
    tol                 = Minimum improvment to be considered significant.
    """
    param_grid = {
        "learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05],
        "n_estimators": [100, 150, 200],
        "max_depth": [2, 3, 4],
    }
    model = GridSearchCV(
        GradientBoostingRegressor(
            validation_fraction=0.1,
            n_iter_no_change=5,
            tol=0.01,
            random_state=42,
        ),
        param_grid,
        cv=2,
    )
    features = df[feature_names].values
    target = df["close"].values
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, target, test_size=0.25, random_state=42
    )

    model.fit(X_train, y_train)
    mse(model, X_test, y_test)
    feature_importance(model, feature_names)
    return model


scaler = StandardScaler()

df = pd.DataFrame(fetch_data())
df = preprocess(df)
model = train(df, scaler)

predictions = []
historical_closes = df["close"].iloc[-20:].tolist()
for _ in range(14):
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

    predicted_close = float(model.best_estimator_.predict(features_future_scaled)[0])
    predictions.append(predicted_close)

    features_future["close"] = predicted_close
    df = pd.concat([df, features_future], ignore_index=True)
    model = train(df, scaler)
    historical_closes.append(predicted_close)

for prediction in predictions:
    print(prediction)

predictions_data = {"predictions": predictions}

prediction_output_dir = "/opt/ml/output/data"
os.makedirs(prediction_output_dir, exist_ok=True)
prediction_file_path = os.path.join(prediction_output_dir, "predictions.csv")
with open(prediction_file_path, "w") as json_file:
    json.dump(predictions_data, json_file, indent=4)

# Save model to S3. SageMaker requires model to be in /opt/ml/model/
model_output_dir = "/opt/ml/model"
model_file_path = os.path.join(model_output_dir, "model.joblib")
os.makedirs(model_output_dir, exist_ok=True)
joblib.dump(model, model_file_path)
