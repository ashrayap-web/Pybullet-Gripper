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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

model = make_pipeline(
    PolynomialFeatures(degree=5),
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred),"\n")
print("Confusion Matrix", confusion_matrix(y_test, y_pred),"\n")
print("Classification Report", classification_report(y_test, y_pred),"\n")