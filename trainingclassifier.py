import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

csv_path = "trainingdata.csv"
out_model = "logistic_model.joblib"
	
df = pd.read_csv(csv_path)

# Features and target
X = df.drop(columns=['Result'])
y = df['Result'].astype(int)

# Basic train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
)

# Pipeline: scaler + logistic regression
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(solver='liblinear', max_iter=1000, random_state=42))
])

print('Training Logistic Regression...')
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")
print('\nClassification report:')
print(classification_report(y_test, y_pred))






