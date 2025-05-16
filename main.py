import os
import traceback
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Load model
model_path = os.path.join(os.path.dirname(__file__), "models", "model.pkl")
try:
    model = joblib.load(model_path)
    print(f"✅ Model loaded from: {model_path}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# Request body schema
class InputData(BaseModel):
    product_type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float
    tool_wear_failure: float = 0.0
    heat_dissipation_failure: float = 0.0
    power_failure: float = 0.0
    overstrain_failure: float = 0.0
    random_failure: float = 0.0

# Helper to encode product type
def encode_product_type(pt: str) -> int:
    mapping = {"typea": 0, "typeb": 1, "typec": 2}
    return mapping.get(pt.lower(), -1)

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    pt_feature = encode_product_type(data.product_type)
    if pt_feature == -1:
        raise HTTPException(status_code=400, detail="Invalid product_type. Use typea, typeb, or typec.")

    try:
        features = np.array([[
            pt_feature,
            data.air_temperature,
            data.process_temperature,
            data.rotational_speed,
            data.torque,
            data.tool_wear,
            data.tool_wear_failure,
            data.heat_dissipation_failure,
            data.power_failure,
            data.overstrain_failure,
            data.random_failure
        ]])

        prediction = model.predict(features)[0]
        confidence = None

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            confidence = round(float(max(proba)), 2)

        result = {
            "prediction": "Failure" if prediction == 1 else "No Failure"
        }

        if confidence is not None:
            result["confidence"] = confidence

        return result

    except Exception:
        return {"error": traceback.format_exc()}
