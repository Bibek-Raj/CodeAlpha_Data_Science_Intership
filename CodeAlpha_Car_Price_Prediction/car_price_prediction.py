# -------------------------------------------
# CodeAlpha Internship - Task 3
# Car Price Prediction using Machine Learning
# -------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------------------
# Load Dataset
# -------------------------------------------

df = pd.read_csv("car data.csv")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

# -------------------------------------------
# Feature Engineering
# -------------------------------------------

# Calculate age of the car
current_year = 2025
df["Car_Age"] = current_year - df["Year"]

# Drop unnecessary columns
df.drop(["Car_Name", "Year"], axis=1, inplace=True)

# -------------------------------------------
# Encode Categorical Variables
# -------------------------------------------

encoder = LabelEncoder()

df["Fuel_Type"] = encoder.fit_transform(df["Fuel_Type"])
df["Selling_type"] = encoder.fit_transform(df["Selling_type"])
df["Transmission"] = encoder.fit_transform(df["Transmission"])

# -------------------------------------------
# Correlation Heatmap
# -------------------------------------------

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# -------------------------------------------
# Prepare Data
# -------------------------------------------

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------------------
# Train Model
# -------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------------------
# Prediction
# -------------------------------------------

y_pred = model.predict(X_test)

# -------------------------------------------
# Evaluation
# -------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R² Score : {r2:.2f}")

# -------------------------------------------
# Actual vs Predicted
# -------------------------------------------

plt.figure(figsize=(7,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Price")

plt.tight_layout()
plt.show()

# -------------------------------------------
# Feature Importance
# -------------------------------------------

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values()

plt.figure(figsize=(8,5))
importance.plot(kind="barh")

plt.title("Feature Importance")

plt.tight_layout()

plt.show()

print("\nProject Completed Successfully!")