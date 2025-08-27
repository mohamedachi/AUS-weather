# AUSWeather Machine Learning Project
# Converted from Jupyter Notebook to Python script

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset (replace with actual CSV if available)
# Example: df = pd.read_csv("weatherAUS.csv")
print("👉 Please load your dataset here (replace with actual file path).")

# For demonstration, let's create a mock dataset
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
df['target'] = y

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("target", axis=1),
    df["target"],
    test_size=0.2,
    random_state=42
)

# ----------------------------
# RandomForest Model
# ----------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(random_state=42))
])

param_grid = {
    "classifier__n_estimators": [50, 100],
    "classifier__max_depth": [None, 10, 20]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy")
model = grid_search.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Random Forest Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ----------------------------
# Exercise 14: Feature Importances
# ----------------------------
feature_importances = grid_search.best_estimator_['classifier'].feature_importances_
importances_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': feature_importances
}).sort_values(by="importance", ascending=False)

print("\nTop Feature Importances (Random Forest):")
print(importances_df.head(10))

# ----------------------------
# Exercise 15: Logistic Regression Model
# ----------------------------
pipeline.set_params(classifier=LogisticRegression(random_state=42))
grid_search.estimator = pipeline

param_grid = {
    'classifier__solver': ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight': [None, 'balanced']
}

grid_search.param_grid = param_grid

model = grid_search.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
