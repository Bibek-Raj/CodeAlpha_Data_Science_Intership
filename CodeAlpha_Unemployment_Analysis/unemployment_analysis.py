# ------------------------------------------
# CodeAlpha Internship - Task 2
# Unemployment Analysis with Python
# ------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------
# Load Dataset
# ------------------------------------------

df = pd.read_csv("Unemployment.csv")

# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# ------------------------------------------
# Basic Information
# ------------------------------------------

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

# ------------------------------------------
# Data Cleaning
# ------------------------------------------

df = df.dropna()

# ------------------------------------------
# Average Unemployment by State
# ------------------------------------------

state_avg = (
    df.groupby("Region")["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Unemployment Rate by State")
print(state_avg)

# ------------------------------------------
# Visualization 1
# State-wise Unemployment
# ------------------------------------------

plt.figure(figsize=(12,8))

sns.barplot(
    x=state_avg.values,
    y=state_avg.index,
)

plt.title("Average Unemployment Rate by State")
plt.xlabel("Average Unemployment Rate (%)")
plt.ylabel("State")

plt.tight_layout()
plt.show()

# ------------------------------------------
# Visualization 2
# Monthly Trend
# ------------------------------------------

monthly = (
    df.groupby(df["Date"].dt.to_period("M"))
    ["Estimated Unemployment Rate (%)"]
    .mean()
)

plt.figure(figsize=(12,6))

monthly.index = monthly.index.astype(str)

plt.plot(
    monthly.index,
    monthly.values,
    marker="o"
)

plt.xticks(rotation=45)

plt.title("Monthly Average Unemployment Rate")

plt.xlabel("Month")

plt.ylabel("Unemployment Rate (%)")

plt.grid(True)

plt.tight_layout()

plt.show()

# ------------------------------------------
# Visualization 3
# Urban vs Rural
# ------------------------------------------

plt.figure(figsize=(6,5))

sns.boxplot(
    data=df,
    x="Area",
    y="Estimated Unemployment Rate (%)"
)

plt.title("Urban vs Rural Unemployment")

plt.tight_layout()

plt.show()

# ------------------------------------------
# Visualization 4
# Correlation Heatmap
# ------------------------------------------

plt.figure(figsize=(8,6))

corr = df[
    [
        "Estimated Unemployment Rate (%)",
        "Estimated Employed",
        "Estimated Labour Participation Rate (%)",
    ]
].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="Blues"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.show()

# ------------------------------------------
# COVID-19 Analysis
# ------------------------------------------

covid = df[df["Date"] >= "2020-03-01"]

covid_avg = covid.groupby("Region")[
    "Estimated Unemployment Rate (%)"
].mean().sort_values(ascending=False)

print("\nTop States During COVID-19")

print(covid_avg.head(10))

# ------------------------------------------
# Insights
# ------------------------------------------

print("\nKey Insights")

print("- States have different unemployment patterns.")

print("- COVID-19 period shows increased unemployment.")

print("- Urban and Rural unemployment differ significantly.")

print("- Labour participation influences employment levels.")

print("\nProject Completed Successfully!")