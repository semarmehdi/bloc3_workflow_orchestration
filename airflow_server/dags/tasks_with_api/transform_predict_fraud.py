import json
import requests
import pandas as pd
from datetime import datetime

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import Variable

# =========================================================
# Schema attendu par le modele (aligne sur train.py)
# =========================================================
MODEL_FEATURES = [
    "merchant",
    "category",
    "amt",
    "gender",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "merch_lat",
    "merch_long",
    "hour",
    "day_of_week",
    "age",
]
NUMERIC_RAW = ["amt", "lat", "long", "city_pop", "merch_lat", "merch_long", "zip"]


def count_fraud(values):
    """Compte les predictions == 1 (robuste int/float/str/None)."""
    n = 0
    for v in values:
        try:
            if int(float(v)) == 1:
                n += 1
        except (TypeError, ValueError):
            continue
    return n


def reconstruct_rows(items):
    rows = []
    for it in items:
        payload = it.get("data")
        if isinstance(payload, str):
            payload = json.loads(payload)

        cols = payload.get("columns")
        data = payload.get("data")
        if not cols or not data:
            continue

        values = data[0] if isinstance(data, list) and len(data) > 0 else []
        rows.append({c: v for c, v in zip(cols, values)})
    return rows


def build_model_features(raw_df, trans_dt):
    """Memes features qu'a l'entrainement (cf train.py)."""
    X = pd.DataFrame(index=raw_df.index)

    for c in ["merchant", "category", "gender", "state", "job"]:
        X[c] = raw_df[c].astype(str)

    for c in NUMERIC_RAW:
        X[c] = pd.to_numeric(raw_df[c], errors="coerce")

    X["hour"] = trans_dt.dt.hour.astype(int)
    X["day_of_week"] = trans_dt.dt.dayofweek.astype(int)

    dob = pd.to_datetime(raw_df["dob"], errors="coerce")
    X["age"] = ((trans_dt - dob).dt.days // 365).astype(int)

    return X[MODEL_FEATURES]


def predict_from_api(predict_url, payload, request_timeout):
    response = requests.post(predict_url, json=payload, timeout=request_timeout)
    response.raise_for_status()
    result = response.json()

    prediction = result.get("prediction")
    proba_0 = result.get("proba_0")
    proba_1 = result.get("proba_1")
    return prediction, proba_0, proba_1


def get_model_endpoint():
    base = Variable.get("FRAUD_MODEL_API_BASE_URL")
    endpoint = Variable.get("FRAUD_MODEL_API_PREDICT_ENDPOINT", default_var="/predict")
    return f"{base.rstrip('/')}/{endpoint.lstrip('/')}"


def predict_with_model_fraud(**context):
    """
    Lit le batch brut S3, reconstruit + feature-engineer les lignes,
    appelle l'API de serving pour chaque transaction (1 appel / ligne),
    sauvegarde les predictions CSV sur S3, push la S3 key via XCom.
    """
    ti = context["task_instance"]

    request_timeout = int(Variable.get("FRAUD_MODEL_API_TIMEOUT", default_var="120"))
    bucket = Variable.get("S3BucketName")
    s3_prefix_predictions = Variable.get("FRAUD_S3_PRED_PREFIX")

    raw_s3_key = ti.xcom_pull(
        task_ids="extract_raw_transactions_batch", key="fraud_raw_s3_key"
    )
    if not raw_s3_key:
        raise ValueError("Missing XCom raw_s3_key (key='fraud_raw_s3_key').")

    s3_hook = S3Hook(aws_conn_id="aws_default")
    local_raw_path = s3_hook.download_file(
        key=raw_s3_key, bucket_name=bucket, local_path="/tmp"
    )
    print(
        f"[INFO] Downloaded raw batch: s3://{bucket}/{raw_s3_key} -> {local_raw_path}"
    )

    with open(local_raw_path, "r") as f:
        raw_batch = json.load(f)

    items = raw_batch.get("items", [])
    if not items:
        raise ValueError("Raw batch contains no items.")

    rows = reconstruct_rows(items)
    if not rows:
        raise ValueError("No usable rows rebuilt from raw batch.")

    raw_df = pd.DataFrame(rows)
    print(f"[INFO] Raw dataframe built: shape={raw_df.shape}")

    trans_dt = pd.to_datetime(pd.to_numeric(raw_df["current_time"]), unit="ms")
    X = build_model_features(raw_df, trans_dt)
    print(f"[INFO] Model features built: shape={X.shape}")

    # Appel API ligne par ligne (1 seul appel par transaction)
    predict_url = get_model_endpoint()

    predictions, proba_0_list, proba_1_list = [], [], []

    for idx, row in X.iterrows():
        payload = json.loads(row.to_json())  # pandas/numpy -> JSON natif
        try:
            prediction, proba_0, proba_1 = predict_from_api(
                predict_url, payload, request_timeout
            )

            if isinstance(prediction, list) and len(prediction) > 0:
                prediction = prediction[0]
            if isinstance(proba_0, list) and len(proba_0) > 0:
                proba_0 = proba_0[0]
            if isinstance(proba_1, list) and len(proba_1) > 0:
                proba_1 = proba_1[0]

            predictions.append(prediction)
            proba_0_list.append(proba_0)
            proba_1_list.append(proba_1)
            print(f"[INFO] Prediction OK for row {idx}")

        except Exception as e:
            print(f"[ERROR] Prediction failed for row {idx}: {e}")
            predictions.append(None)
            proba_0_list.append(None)
            proba_1_list.append(None)

    # Sortie : colonnes brutes (sans current_time) + trans_time + resultats
    result = raw_df.drop(columns=["current_time"]).copy()
    result["trans_time"] = trans_dt.astype(str).values
    result["prediction"] = predictions
    result["proba_0"] = proba_0_list
    result["proba_1"] = proba_1_list

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    predictions_filename = f"{ts}_fraud_predictions.csv"
    local_results = f"/tmp/{predictions_filename}"
    result.to_csv(local_results, index=False)

    s3_key_predictions = f"{s3_prefix_predictions}/{predictions_filename}"
    s3_hook.load_file(
        filename=local_results, key=s3_key_predictions, bucket_name=bucket, replace=True
    )

    n_fraud = count_fraud(result["prediction"])
    ti.xcom_push(key="fraud_predictions_s3_key", value=s3_key_predictions)
    ti.xcom_push(key="fraud_predictions_count", value=len(result))
    ti.xcom_push(key="fraud_detected_count", value=n_fraud)

    print(
        f"[INFO] Predictions saved: s3://{bucket}/{s3_key_predictions} "
        f"(rows={len(result)}, fraud={n_fraud})"
    )
