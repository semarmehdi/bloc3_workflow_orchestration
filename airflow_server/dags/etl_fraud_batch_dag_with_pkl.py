from datetime import datetime, timedelta

from airflow import DAG

# Airflow 2.x : PythonOperator vit ici (airflow.providers.standard = Airflow 3 only)
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

# Airflow 2.x : on passe par common.sql (PostgresOperator est deprecated)
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from operators.s3_to_postgres import S3ToPostgresOperator

from tasks_with_pkl.extract_fraud import extract_transactions_batch_to_s3
from tasks_with_pkl.load_model_fraud import load_model_task
from tasks_with_pkl.transform_predict_fraud import predict_with_model_fraud
from tasks_with_pkl.notify_fraud import send_fraud_notification


default_args = {
    "owner": "mehdi",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="etl_fraud_batch_pkl_dag",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["demo", "etl", "batch", "pkl", "fraud"],
) as dag:

    # =========================
    # 1) Extract
    # =========================
    extract_task = PythonOperator(
        task_id="extract_raw_transactions_batch",
        python_callable=extract_transactions_batch_to_s3,
    )

    # =========================
    # 2) Load model (en parallele de l'extract)
    # =========================
    load_model = PythonOperator(
        task_id="load_model",
        python_callable=load_model_task,
    )

    # =========================
    # 3) Transform / Predict (attend extract + model)
    # =========================
    transform = PythonOperator(
        task_id="predict_with_model_fraud",
        python_callable=predict_with_model_fraud,
    )

    # extract + load_model en parallele, puis transform (branche unique)
    [extract_task, load_model] >> transform

    # =========================
    # 4) LOAD (Postgres)
    # =========================
    with TaskGroup(group_id="load_branch") as load_branch:

        create_predictions_table = SQLExecuteQueryOperator(
            task_id="create_predictions_table",
            sql="""
            CREATE TABLE IF NOT EXISTS fraud_predictions (
                id SERIAL PRIMARY KEY,
                cc_num TEXT,
                merchant TEXT,
                category TEXT,
                amt TEXT,
                first TEXT,
                last TEXT,
                gender TEXT,
                street TEXT,
                city TEXT,
                state TEXT,
                zip TEXT,
                lat TEXT,
                long TEXT,
                city_pop TEXT,
                job TEXT,
                dob TEXT,
                trans_num TEXT,
                merch_lat TEXT,
                merch_long TEXT,
                is_fraud TEXT,
                trans_time TEXT,
                prediction TEXT,
                proba_0 TEXT,
                proba_1 TEXT
            );
            """,
            conn_id="postgres_default",
        )

        transfer_predictions_to_postgres = S3ToPostgresOperator(
            task_id="transfer_predictions_to_postgres",
            table="fraud_predictions",
            bucket="{{ var.value.S3BucketName }}",
            key="{{ task_instance.xcom_pull(task_ids='predict_with_model_fraud', key='fraud_predictions_s3_key') }}",
            postgres_conn_id="postgres_default",
            aws_conn_id="aws_default",
        )

        create_predictions_table >> transfer_predictions_to_postgres

    # =========================
    # 5) NOTIFY (email) en parallele du load, juste apres predict
    #    batch -> notify_on_no_fraud=True (mail [FRAUD ALERT] ou [RAS])
    # =========================
    notify = PythonOperator(
        task_id="notify_fraud",
        python_callable=send_fraud_notification,
        op_kwargs={"notify_on_no_fraud": True},
    )

    # transform puis, en parallele : load_branch + notification
    transform >> [load_branch, notify]
