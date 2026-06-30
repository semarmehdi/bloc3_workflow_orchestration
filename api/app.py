import os
import mlflow
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union
from dotenv import load_dotenv


load_dotenv()

# -----------------------------------------------------------------------------
# ENV + MLflow setup
# -----------------------------------------------------------------------------
# Sur Hugging Face, ces variables sont lues depuis les "Secrets"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME")  # ex: "fraud_detection"
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS")  # ex: "challenger"

# On force l'URI pour mlflow
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def build_model_uri() -> str:
    if MODEL_ALIAS:
        return f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    return f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE or 'Production'}"
    # ex: models:/fraud_detection@challenger


MODEL_URI = build_model_uri()
MODEL = None

# -----------------------------------------------------------------------------
# FastAPI Setup
# -----------------------------------------------------------------------------
app = FastAPI(title="💳 Fraud Detection API")


# -----------------------------------------------------------------------------
# Schema d'entree = les 15 features attendues par le modele (cf train.py)
# X = df.drop(cols_to_drop), avec hour / day_of_week / age calcules en amont
# (dans la task Airflow). L'API recoit donc deja les features pretes.
# -----------------------------------------------------------------------------
class PredictionFeatures(BaseModel):
    merchant: str
    category: str
    amt: Union[int, float]
    gender: str
    state: str
    zip: Union[int, float]
    lat: Union[int, float]
    long: Union[int, float]
    city_pop: Union[int, float]
    job: str
    merch_lat: Union[int, float]
    merch_long: Union[int, float]
    hour: int
    day_of_week: int
    age: int


# -----------------------------------------------------------------------------
# Startup: CHARGEMENT BLOQUANT (Solution au bug 500)
# -----------------------------------------------------------------------------
@app.on_event("startup")
def load_model_sync():
    global MODEL
    print(f"🚀 [INFO] Attempting to load model: {MODEL_URI}")
    try:
        # On attend que le chargement soit fini avant de rendre l'API disponible
        # flaveur sklearn -> Pipeline imblearn complete (predict + predict_proba)
        MODEL = mlflow.sklearn.load_model(MODEL_URI)
        print("✅ [INFO] Model loaded successfully!")
    except Exception as e:
        print(f"❌ [ERROR] Failed to load model: {e}")
        # En cas d'échec, on laisse MODEL à None pour que /health le signale


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_uri": MODEL_URI,
        "model_loaded": MODEL is not None,
    }


@app.post("/predict")
async def predict(payload: PredictionFeatures):
    if MODEL is None:
        raise HTTPException(
            status_code=503, detail="Model is still loading or failed to load."
        )

    # Conversion pydantic -> dict -> DataFrame (1 ligne)
    df = pd.DataFrame([payload.dict()])

    pred = MODEL.predict(df)

    proba_0 = None
    proba_1 = None
    try:
        proba = MODEL.predict_proba(df)
        proba_0 = float(proba[0][0])
        proba_1 = float(proba[0][1])
    except Exception as e:
        print(f"⚠️ [WARN] predict_proba unavailable: {e}")

    return {
        "prediction": int(pred[0]),
        "proba_0": proba_0,
        "proba_1": proba_1,
    }


if __name__ == "__main__":
    # Port 7860 est le standard pour Hugging Face Spaces
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
