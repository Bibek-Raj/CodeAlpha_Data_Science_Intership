# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset

df = pd.read_csv("Iris.csv")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nSpecies Count")
print(df["Species"].value_counts())

# Drop ID Column

df = df.drop("Id", axis=1)

# Data Visualization

sns.pairplot(df, hue="Species")
plt.suptitle("Iris Flower Pair Plot", y=1.02)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.drop("Species", axis=1).corr(), annot=True, cmap="Blues")
plt.title("Feature Correlation")
plt.show()

# Prepare Data

X = df.drop("Species", axis=1)
y = df["Species"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)

# Evaluation

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("\nPrediction Example")

sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print("Predicted Species:", prediction[0])

print("\nProject Completed Successfully!")