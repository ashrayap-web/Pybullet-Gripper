import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


df = pd.read_csv("arm_reachability_down.csv")

success_data = df.loc[df["Result"]==1.0]
fail_data = df.loc[df["Result"]==0.0]

# visualise data
plt.figure(figsize=(10,8))
plt.scatter(success_data["x"], success_data["y"], c='green', marker='o', s=20, label='success')
plt.scatter(fail_data["x"], fail_data["y"], c='red', marker='x', s=20, label='failure')
plt.show()

# preprocessing
X = df.drop(columns=["Result"])
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = make_pipeline(
    PolynomialFeatures(),
    StandardScaler(),
    LogisticRegression(penalty="l2", max_iter=10000)  # lower C -> stricter regularization
)

param_grid = {
    'polynomialfeatures__degree': [7, 8, 9, 10],
    'logisticregression__C': [10.0, 100.0, 1000.0, 10000.0]
}

# uses val it creates by itself
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# train
grid_search.fit(X_train, y_train)
print("Best params:", grid_search.best_params_)

model = grid_search.best_estimator_

# test
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("\nTrain Accuracy:", accuracy_score(y_train, y_train_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred),"\n")

print("Confusion Matrix", confusion_matrix(y_test, y_test_pred),"\n")
print("Classification Report", classification_report(y_test, y_test_pred),"\n")


#----------------------- DECISION BOUNDARY ---------------#
X_vals = X.values

x_min, x_max = X_vals[:, 0].min() - 0.05, X_vals[:, 0].max() + 0.05
y_min, y_max = X_vals[:, 1].min() - 0.05, X_vals[:, 1].max() + 0.05
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

grid = np.c_[xx.ravel(), yy.ravel()]
grid_df = pd.DataFrame(grid, columns=X.columns)
Z = model.predict_proba(grid_df)[:, 1].reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
plt.scatter(success_data["x"], success_data["y"], c='green', marker='o', s=20, label='Success')
plt.scatter(fail_data["x"], fail_data["y"], c='red', marker='x', s=20, label='Failure')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.show()
