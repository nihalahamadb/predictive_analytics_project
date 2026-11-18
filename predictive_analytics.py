import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# STEP 1: LOAD HISTORICAL DATA
# ==========================================
# Loads the included historical dataset (with intentional missing values & duplicates to demonstrate cleaning)
raw_df = pd.read_csv("historical_sales_data.csv")

print("--- Raw Data Summary ---")
print(f"Total rows: {len(raw_df)}")
print(f"Missing values:\n{raw_df.isnull().sum()}\n")

# ==========================================
# STEP 2: CLEAN & PREPROCESS DATASET
# ==========================================
df = raw_df.copy()

# 1. Convert Date column to datetime and sort chronologically
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

# 2. Remove duplicate dates
df = df.drop_duplicates(subset=["Date"], keep="first").reset_index(drop=True)

# 3. Handle missing target values using time-based linear interpolation
df["Sales"] = df["Sales"].interpolate(method="linear").bfill().ffill()

# Save cleaned dataset
df.to_csv("cleaned_sales_data.csv", index=False)

# 4. Feature Engineering (Extract time-series features for Regression)
def add_time_features(data, start_date):
    d = data.copy()
    d["Days_Since_Start"] = (d["Date"] - start_date).dt.days
    d["Day_Of_Week"] = d["Date"].dt.dayofweek
    d["Month"] = d["Date"].dt.month
    d["Day_Of_Year"] = d["Date"].dt.dayofyear
    # Cyclical encoding so the model understands Dec 31 is right next to Jan 1
    d["Sin_Year"] = np.sin(2 * np.pi * d["Day_Of_Year"] / 365.25)
    d["Cos_Year"] = np.cos(2 * np.pi * d["Day_Of_Year"] / 365.25)
    d["Is_Weekend"] = (d["Day_Of_Week"] >= 5).astype(int)
    return d

start_date = df["Date"].min()
df_features = add_time_features(df, start_date)

feature_cols = ["Days_Since_Start", "Day_Of_Week", "Month", "Sin_Year", "Cos_Year", "Is_Weekend"]
X = df_features[feature_cols]
y = df_features["Sales"]

# ==========================================
# STEP 3: TRAIN / TEST SPLIT & MODEL TRAINING
# ==========================================
# Time-series rule: Never shuffle! Train on the past (80%), test on the recent future (20%)
split_index = int(len(df_features) * 0.8)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
dates_train, dates_test = df_features["Date"].iloc[:split_index], df_features["Date"].iloc[split_index:]

# Ridge Regression prevents overfitting while capturing linear trends + seasonality
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# ==========================================
# STEP 4: EVALUATE MODEL ACCURACY
# ==========================================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("--- Model Evaluation Metrics (Test Set) ---")
print(f"MAE  (Mean Absolute Error):      {mae:.2f}")
print(f"RMSE (Root Mean Squared Error):  {rmse:.2f}")
print(f"R2   (Variance Explained):       {r2:.4f}")
print(f"MAPE (Mean Abs % Error):         {mape:.2f}%\n")

# ==========================================
# STEP 5: FORECAST FUTURE TRENDS (NEXT 180 DAYS)
# ==========================================
# Retrain on 100% of historical data before forecasting into the unknown future
model.fit(X, y)

last_date = df["Date"].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=180, freq="D")
future_df = pd.DataFrame({"Date": future_dates})
future_features = add_time_features(future_df, start_date)

future_df["Predicted_Sales"] = np.round(model.predict(future_features[feature_cols]), 2)

# Save forecast to CSV for your project submission
future_df.to_csv("future_sales_forecast.csv", index=False)
print("--- Sample Future Forecast (Next 5 Days) ---")
print(future_df.head())
print("\nSaved full 180-day forecast to 'future_sales_forecast.csv'")

# ==========================================
# STEP 6: VISUALIZE HISTORICAL, TEST & FORECAST
# ==========================================
plt.figure(figsize=(14, 6))

# Plot actual historical data
plt.plot(dates_train, y_train, label="Historical Data (Train)", color="#1f77b4", alpha=0.7)
plt.plot(dates_test, y_test, label="Actual Data (Test)", color="#2ca02c", alpha=0.85)

# Plot test predictions to show accuracy
plt.plot(dates_test, y_pred, label="Model Validation (Test Pred)", color="#ff7f0e", linestyle="--", linewidth=2)

# Plot future 180-day forecast
plt.plot(future_df["Date"], future_df["Predicted_Sales"], label="180-Day Future Forecast", color="#d62728", linewidth=2)

plt.title("Predictive Analytics: Historical Trends & 180-Day Sales Forecast", fontsize=14, fontweight="bold")
plt.xlabel("Date", fontsize=12)
plt.ylabel("Sales ($)", fontsize=12)
plt.legend(loc="upper left")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()

# Save chart as high-res image for your presentation/report
plt.savefig("forecast_visualization.png", dpi=300)
print("Saved chart to 'forecast_visualization.png'")
plt.show()
