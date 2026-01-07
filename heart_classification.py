# heart_classification.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv("heart.csv")

print("First 5 rows:")
print(df.head())

#simple feature set 
selected_features = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "MaxHR",
    "ExerciseAngina",
    "ST_Slope",
]

categorical = [col for col in selected_features if df[col].dtype == "object"]
numerical = [col for col in selected_features if df[col].dtype != "object"]

# relationship between categorical features and heart disease
for col in categorical:
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x=col, y="HeartDisease", errorbar=None)
    plt.title(f"{col} vs HeartDisease")
    plt.ylabel("Mean target (0=no, 1=yes)")
    plt.tight_layout()
    plt.show()

# relationship between numerical features and heart disease
for col in numerical:
    plt.figure(figsize=(6, 4))
    sns.regplot(data=df, x=col, y="HeartDisease", logistic=True, ci=None)
    plt.title(f"{col} vs HeartDisease")
    plt.ylabel("Estimated probability")
    plt.tight_layout()
    plt.show()

# pairplot for key variables 
sns.pairplot(df[["Age", "RestingBP", "MaxHR", "HeartDisease"]], hue="HeartDisease", diag_kind="kde")
plt.suptitle("Selected feature relationships", y=1.02)
plt.show()

# model input
X = pd.get_dummies(df[selected_features], drop_first=True)
y = df["HeartDisease"].astype(int)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# baseline model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# evaluation
y_pred = model.predict(X_test_scaled)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred))

# coefficients
importance = pd.Series(model.coef_[0], index=X.columns)
importance.sort_values().plot(kind="barh", title="Logistic Regression Coefficients")

plt.xlabel("Coefficient")
plt.tight_layout()
plt.show()