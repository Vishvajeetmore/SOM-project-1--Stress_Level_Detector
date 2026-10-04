"""POST /predict — returns stress label, probabilities, and SHAP values."""
import os
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")
pipeline = joblib.load(MODEL_PATH)

FEATURES = [
    "Study_Hours", "Hobbies_Hours", "Sleep_Hours",
    "Social_Interaction_Hours", "Physical_Activity_Hours", "CGPA",
]
LABELS = {0: "High", 1: "Moderate", 2: "Low"}

# Background dataset for SHAP KernelExplainer (small sample keeps it fast)
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "student_lifestyle_dataset2.csv")
_df = pd.read_csv(DATA_PATH)
_background = _df[FEATURES].sample(50, random_state=42)


def _predict_proba_fn(x):
    return pipeline.predict_proba(pd.DataFrame(x, columns=FEATURES))


_explainer = shap.KernelExplainer(_predict_proba_fn, _background)


class PredictRequest(BaseModel):
    study_hours: float = Field(..., ge=0, le=24)
    hobbies_hours: float = Field(..., ge=0, le=24)
    sleep_hours: float = Field(..., ge=0, le=24)
    social_hours: float = Field(..., ge=0, le=24)
    physical_hours: float = Field(..., ge=0, le=24)
    cgpa: float = Field(..., ge=0, le=4.0)


@router.post("/predict")
def predict(req: PredictRequest):
    row = pd.DataFrame([[
        req.study_hours, req.hobbies_hours, req.sleep_hours,
        req.social_hours, req.physical_hours, req.cgpa,
    ]], columns=FEATURES)

    pred = int(pipeline.predict(row)[0])
    proba = pipeline.predict_proba(row)[0].tolist()

    shap_values = _explainer.shap_values(row.values, nsamples=100)
    # KernelExplainer returns list per class -> take the predicted class
    class_shap = np.asarray(shap_values)
    if class_shap.ndim == 3:
        class_shap = class_shap[0, :, pred]
    else:
        class_shap = class_shap[pred][0]

    return {
        "stress_level": LABELS[pred],
        "probabilities": {LABELS[i]: round(p, 4) for i, p in enumerate(proba)},
        "shap": {f: round(float(v), 4) for f, v in zip(FEATURES, class_shap)},
    }
