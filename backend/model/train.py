"""Train the final stress-level model: SVM (RBF kernel).

Key ideas:
- SVMs are sensitive to feature scale, so we wrap StandardScaler + SVC
  into one sklearn Pipeline. The SAME scaling is then applied automatically
  at prediction time (this avoids a very common bug).
- SVC(probability=True) gives us predict_proba, handy for the UI.
- We evaluate with accuracy + a 5-fold cross-validation score.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES = [
    "Study_Hours", "Hobbies_Hours", "Sleep_Hours",
    "Social_Interaction_Hours", "Physical_Activity_Hours", "CGPA",
]
TARGET_MAP = {"High": 0, "Moderate": 1, "Low": 2}
INV_TARGET_MAP = {v: k for k, v in TARGET_MAP.items()}

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "student_lifestyle_dataset2.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


def train():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["Stress_Level"].map(TARGET_MAP)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cv = cross_val_score(pipeline, X, y, cv=5).mean()

    print(f"Test accuracy : {acc:.4f}")
    print(f"5-fold CV acc : {cv:.4f}")
    print(classification_report(y_test, preds, target_names=["High", "Moderate", "Low"]))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    train()
