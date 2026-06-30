import json
import time
import logging
from datetime import datetime

import requests
from airflow.models import Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook


def get_data(url):
    """
    L'API renvoie une transaction au format .to_json(orient="split") de Pandas.
    Le body est une chaine JSON-encodee (double-encodage) -> on decode 2x si besoin.
    """
    r = requests.get(url, timeout=30, headers={"accept": "application/json"})
    r.raise_for_status()
    payload = json.loads(r.text)
    data = json.loads(payload) if isinstance(payload, str) else payload
    return data


def extract_transactions_batch_to_s3(**context):
    """
    Appelle /current-transactions N fois (defaut 50), collecte les payloads bruts,
    sauvegarde en /tmp en JSON, upload vers S3, push la S3 key via XCom.
    """

    base_url = Variable.get("FRAUD_BASE_URL")
    endpoint = Variable.get("FRAUD_ENDPOINT", default_var="/current-transactions")

    batch_size = int(Variable.get("FRAUD_BATCH_SIZE", default_var="1"))
    sleep_seconds = float(Variable.get("FRAUD_SLEEP_SECONDS", default_var="0.5"))

    bucket = Variable.get("S3BucketName")
    s3_prefix = Variable.get("FRAUD_S3_PREFIX")

    url = f"{base_url}{endpoint}"

    logging.info(
        f"Starting extract batch: size={batch_size}, sleep={sleep_seconds}s, url={url}"
    )

    items = []
    errors = 0

    for i in range(batch_size):
        try:
            data = get_data(url)

            items.append(
                {
                    "pulled_at_utc": datetime.utcnow().isoformat(),
                    "source_url": url,
                    "data": data,
                }
            )
            logging.info(f"Pulled {i + 1}/{batch_size}")

        except Exception as e:
            errors += 1
            logging.warning(f"Error on pull {i + 1}/{batch_size}: {e}", exc_info=True)

        if i < batch_size - 1 and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    filename = (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_fraud_transactions_batch.json"
    )
    local_path = f"/tmp/{filename}"

    artifact = {
        "meta": {
            "batch_size_requested": batch_size,
            "records_collected": len(items),
            "errors": errors,
            "base_url": base_url,
            "endpoint": endpoint,
            "created_at_utc": datetime.utcnow().isoformat(),
        },
        "items": items,
    }

    with open(local_path, "w") as f:
        json.dump(artifact, f)

    s3_key = f"{s3_prefix}/{filename}"
    s3_hook = S3Hook(aws_conn_id="aws_default")
    s3_hook.load_file(
        filename=local_path,
        key=s3_key,
        bucket_name=bucket,
        replace=True,
    )

    ti = context["task_instance"]
    ti.xcom_push(key="fraud_raw_s3_key", value=s3_key)
    ti.xcom_push(key="fraud_records_collected", value=len(items))

    logging.info(
        f"Saved batch to s3://{bucket}/{s3_key} (records={len(items)}, errors={errors})"
    )
