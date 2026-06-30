import mlflow
import boto3
import pickle
from airflow.models import Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook


def load_model_task(**context):
    """
    Charge le modele depuis le MLflow Registry et le sauvegarde en pickle.
    On utilise la flaveur sklearn (et pas pyfunc) pour recuperer la vraie
    Pipeline imblearn -> on garde ainsi acces a predict_proba (proba_1 = fraude).
    Push le chemin du pickle via XCom.
    """

    MLFLOW_TRACKING_URI = Variable.get("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"[INFO] Using MLflow Tracking URI: {MLFLOW_TRACKING_URI}")

    try:
        s3_hook = S3Hook(aws_conn_id="aws_default")
        creds = s3_hook.get_credentials()
        print("[INFO] Found AWS creds from Airflow Connection (aws_default).")
        boto3.setup_default_session(
            aws_access_key_id=creds.access_key,
            aws_secret_access_key=creds.secret_key,
            region_name=Variable.get("AWS_DEFAULT_REGION", default_var="eu-north-1"),
        )
    except Exception as e:
        print(f"[WARN] Could not use Airflow connection aws_default: {e}")
        aws_access_key_id = Variable.get("AWS_ACCESS_KEY_ID", default_var=None)
        aws_secret_access_key = Variable.get("AWS_SECRET_ACCESS_KEY", default_var=None)
        aws_region = Variable.get("AWS_DEFAULT_REGION", default_var="eu-north-1")
        boto3.setup_default_session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region,
        )

    REGISTERED_MODEL_NAME = Variable.get("REGISTERED_MODEL_NAME")  # ex: "fraud_detection"
    ALIAS = Variable.get("ALIAS")  # ex: "challenger"
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{ALIAS}"
    print(f"[INFO] Loading model (sklearn flavor) from {model_uri}")

    # flaveur sklearn -> objet Pipeline imblearn complet (predict + predict_proba)
    model = mlflow.sklearn.load_model(model_uri)

    local_pickle = f"/tmp/{REGISTERED_MODEL_NAME}.pkl"
    with open(local_pickle, "wb") as f:
        pickle.dump(model, f)
    print(f"[INFO] Model pickled at {local_pickle}")

    context["task_instance"].xcom_push(key="model_pickle_path", value=local_pickle)
