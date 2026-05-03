# 1. IMPORT LIBRARIES 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# 2. LOAD DATASET 
df = pd.read_csv("electricity_data.csv")
print("Shape:", df.shape)
print("First 5 rows:\n", df.head())
print("\nData Info:")
df.info()

# 3. DATA PREPROCESSING 
df = df.dropna()                          # Remove missing values
print("\nAfter cleaning:", df.shape)
print("\nBasic Stats:\n", df.describe())

# 4. FEATURE & TARGET SELECTION 
target = "Consumption_kWh"
X = df.drop(target, axis=1)
y = df[target]
print("\nFeatures:", list(X.columns))
print("Target:", target)

#  5. TRAIN-TEST SPLIT (80:20) 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("\nTrain size:", X_train.shape, " Test size:", X_test.shape)

# 6. MODEL TRAINING 
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

#  7. PREDICTION 
y_pred = model.predict(X_test)

#  8. EVALUATION 
print("\n===== Model Performance =====")
print("MAE: ", metrics.mean_absolute_error(y_test, y_pred))
print("MSE: ", metrics.mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2:  ", metrics.r2_score(y_test, y_pred))

# 9. FIG 1: HOURLY TREND LINE 
grouped = df.groupby("Hour")["Consumption_kWh"].mean()
plt.figure(figsize=(8, 4.5))
plt.plot(grouped.index, grouped.values, marker="o", color="#2E75B6", linewidth=2)
plt.fill_between(grouped.index, grouped.values, alpha=0.15, color="#2E75B6")
plt.xlabel("Hour of Day")
plt.ylabel("Avg Consumption (kWh)")
plt.title("Fig 1: Average Electricity Consumption by Hour")
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()

# 10. FIG 2: DAY-WISE BAR CHART 
day_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
day_data = df.groupby("Day_of_Week")["Consumption_kWh"].mean()
colors = ["#C00000" if i >= 5 else "#2E75B6" for i in day_data.index]
plt.figure(figsize=(8, 4.5))
plt.bar(day_labels, day_data.values, color=colors, edgecolor="white")
plt.xlabel("Day of Week")
plt.ylabel("Avg Consumption (kWh)")
plt.title("Fig 2: Average Electricity Consumption by Day of Week")
plt.show()

#11. FIG 3: HISTOGRAM 
plt.figure(figsize=(8, 4.5))
plt.hist(df["Consumption_kWh"], bins=20, color="#70AD47", edgecolor="white")
plt.axvline(df["Consumption_kWh"].mean(), color="red", linestyle="--", lw=2)
plt.xlabel("Consumption (kWh)")
plt.ylabel("Frequency")
plt.title("Fig 3: Distribution of Electricity Consumption")
plt.show()

# 12. FIG 4: ACTUAL vs PREDICTED SCATTER 
plt.figure(figsize=(8, 4.5))
plt.scatter(y_test, y_pred, alpha=0.5, color="#2E75B6", edgecolors="white", s=50)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual Consumption (kWh)")
plt.ylabel("Predicted Consumption (kWh)")
plt.title("Fig 4: Actual vs Predicted Electricity Consumption")
plt.show()

# 13. FIG 5: LINE COMPARISON (40 SAMPLES) 
plt.figure(figsize=(9, 4.5))
plt.plot(range(40), list(y_test[:40]), "b-o", markersize=4, label="Actual")
plt.plot(range(40), list(y_pred[:40]), "r--s", markersize=4, label="Predicted")
plt.xlabel("Sample Index")
plt.ylabel("Consumption (kWh)")
plt.title("Fig 5: Actual vs Predicted Consumption (40 Samples)")
plt.legend()
plt.show()

