import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



df = pd.read_csv("poses_dataset.csv")

min_n = df["Result"].value_counts().min()
print(min_n)

balanced_df = df.groupby("Result", group_keys=False).sample(n=min_n, random_state=42)

X = balanced_df.drop(columns=["Result"])
#print(X.head())

y = balanced_df["Result"]
#print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = make_pipeline(
    PolynomialFeatures(degree=6),
    StandardScaler(),
    LogisticRegression(penalty="l2", C=10.0, max_iter=10000)  # lower C -> stricter regularization
)

# train
model.fit(X_train, y_train)


# test
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Train Accuracy:", accuracy_score(y_train, y_train_pred),"\n")
print("Test Accuracy:", accuracy_score(y_test, y_test_pred),"\n")

print("Confusion Matrix", confusion_matrix(y_test, y_test_pred),"\n")
print("Classification Report", classification_report(y_test, y_test_pred),"\n")