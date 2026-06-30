import os
import json
import pickle
import pandas as pd
from datetime import datetime

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import Variable


# =========================================================
# Schema attendu par le modele (aligne sur train.py)
# X = df.drop(cols_to_drop) -> 15 colonnes, dont 3 calculees
# =========================================================
MODEL_FEATURES = [
    "merchant", "category", "amt", "gender", "state", "zip",
    "lat", "long", "city_pop", "job", "merch_lat", "merch_long",
    "hour", "day_of_week", "age",
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
    """Reconstruit les lignes brutes depuis les payloads format split."""
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
    """
    Reproduit le feature engineering de train.py a l'inference :
      - hour / day_of_week depuis current_time (epoch ms)
      - age = (current_time - dob) en annees
    Retourne un DataFrame avec EXACTEMENT les MODEL_FEATURES, dans l'ordre.
    """
    X = pd.DataFrame(index=raw_df.index)

    # categoricals -> str
    for c in ["merchant", "category", "gender", "state", "job"]:
        X[c] = raw_df[c].astype(str)

    # numeriques bruts -> numeric robuste
    for c in NUMERIC_RAW:
        X[c] = pd.to_numeric(raw_df[c], errors="coerce")

    # features temporelles (current_time = epoch ms)
    X["hour"] = trans_dt.dt.hour.astype(int)
    X["day_of_week"] = trans_dt.dt.dayofweek.astype(int)

    dob = pd.to_datetime(raw_df["dob"], errors="coerce")
    X["age"] = ((trans_dt - dob).dt.days // 365).astype(int)

    return X[MODEL_FEATURES]


def predict_with_model_fraud(**context):
    ti = context["task_instance"]

    # 1) Modele pickle (task load_model en parallele)
    pickle_path = ti.xcom_pull(task_ids="load_model", key="model_pickle_path")
    if not pickle_path or not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle model path not found: {pickle_path}")

    print(f"[INFO] Using pickled model at {pickle_path}")
    with open(pickle_path, "rb") as f:
        model = pickle.load(f)

    # 2) Batch brut depuis S3
    raw_s3_key = ti.xcom_pull(
        task_ids="extract_raw_transactions_batch", key="fraud_raw_s3_key"
    )
    if not raw_s3_key:
        raise ValueError("Missing XCom raw_s3_key (key='fraud_raw_s3_key').")

    bucket = Variable.get("S3BucketName")
    s3_hook = S3Hook(aws_conn_id="aws_default")

    local_raw_path = s3_hook.download_file(
        key=raw_s3_key, bucket_name=bucket, local_path="/tmp"
    )
    print(f"[INFO] Downloaded raw batch: s3://{bucket}/{raw_s3_key} -> {local_raw_path}")

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

    # 3) Feature engineering (current_time en ms -> datetime)
    trans_dt = pd.to_datetime(pd.to_numeric(raw_df["current_time"]), unit="ms")
    X = build_model_features(raw_df, trans_dt)
    print(f"[INFO] Model features built: shape={X.shape}")

    # 4) Prediction locale (sklearn flavor -> predict + predict_proba)
    preds = model.predict(X)

    probs = None
    try:
        probs = model.predict_proba(X)
    except Exception as e:
        print(f"[WARN] predict_proba unavailable: {e}")
        probs = None

    # 5) Sortie : colonnes brutes (sans current_time) + trans_time lisible + resultats
    result = raw_df.drop(columns=["current_time"]).copy()
    result["trans_time"] = trans_dt.astype(str).values
    result["prediction"] = preds

    if probs is not None:
        try:
            result["proba_0"] = [p[0] for p in probs]
            result["proba_1"] = [p[1] for p in probs]
        except Exception:
            result["proba_0"] = None
            result["proba_1"] = None
    else:
        result["proba_0"] = None
        result["proba_1"] = None

    # 6) Sauvegarde CSV + upload S3
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    predictions_filename = f"{ts}_fraud_predictions.csv"
    local_results = f"/tmp/{predictions_filename}"
    result.to_csv(local_results, index=False)

    s3_prefix_predictions = Variable.get("FRAUD_S3_PRED_PREFIX")
    s3_key_predictions = f"{s3_prefix_predictions}/{predictions_filename}"

    s3_hook.load_file(
        filename=local_results, key=s3_key_predictions, bucket_name=bucket, replace=True
    )

    # 7) XComs
    n_fraud = count_fraud(result["prediction"])
    ti.xcom_push(key="fraud_predictions_s3_key", value=s3_key_predictions)
    ti.xcom_push(key="fraud_predictions_count", value=len(result))
    ti.xcom_push(key="fraud_detected_count", value=n_fraud)

    print(
        f"[INFO] Predictions saved: s3://{bucket}/{s3_key_predictions} "
        f"(rows={len(result)}, fraud={n_fraud})"
    )
