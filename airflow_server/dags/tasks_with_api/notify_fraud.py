import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

from airflow.models import Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook


def _send_email(subject, body, attachment_path=None, attachment_name=None):
    """Envoi SMTP Gmail (reprise du modele test_smtp.py, parametre via Variables)."""
    sender = Variable.get("SMTP_SENDER_EMAIL")
    app_password = Variable.get(
        "SMTP_APP_PASSWORD"
    )  # cle d'application Google (16 car.)
    receiver = Variable.get("SMTP_RECEIVER_EMAIL")
    host = Variable.get("SMTP_HOST", default_var="smtp.gmail.com")
    port = int(Variable.get("SMTP_PORT", default_var="587"))

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    if attachment_path and os.path.exists(attachment_path):
        name = attachment_name or os.path.basename(attachment_path)
        with open(attachment_path, "rb") as f:
            part = MIMEApplication(f.read(), Name=name)
        part["Content-Disposition"] = f'attachment; filename="{name}"'
        msg.attach(part)

    server = smtplib.SMTP(host, port)
    try:
        server.starttls()
        server.login(sender, app_password)
        server.sendmail(sender, receiver, msg.as_string())
    finally:
        server.quit()


def send_fraud_notification(notify_on_no_fraud=True, **context):
    """
    Envoie une notification email selon le resultat du batch de predictions.

    - notify_on_no_fraud=True  (BATCH) : on envoie toujours un mail
        -> [FRAUD ALERT] si au moins 1 fraude, sinon [RAS]
    - notify_on_no_fraud=False (UNITAIRE / STREAM) : mail UNIQUEMENT s'il y a fraude
    """
    ti = context["task_instance"]

    n_fraud = int(
        ti.xcom_pull(task_ids="predict_with_model_fraud", key="fraud_detected_count")
        or 0
    )
    n_total = int(
        ti.xcom_pull(task_ids="predict_with_model_fraud", key="fraud_predictions_count")
        or 0
    )
    s3_key = ti.xcom_pull(
        task_ids="predict_with_model_fraud", key="fraud_predictions_s3_key"
    )

    # cas unitaire / stream : silence radio si aucune fraude
    if n_fraud == 0 and not notify_on_no_fraud:
        print("[INFO] Aucune fraude detectee -> pas de notification (mode unitaire).")
        return

    # telechargement du CSV de predictions pour la piece jointe
    attachment_path = None
    attachment_name = None
    if s3_key:
        bucket = Variable.get("S3BucketName")
        s3_hook = S3Hook(aws_conn_id="aws_default")
        attachment_path = s3_hook.download_file(
            key=s3_key, bucket_name=bucket, local_path="/tmp"
        )
        attachment_name = os.path.basename(s3_key)

    if n_fraud > 0:
        subject = f"[FRAUD ALERT] {n_fraud} transaction(s) suspecte(s) sur {n_total}"
        body = (
            "Bonjour,\n\n"
            f"{n_fraud} transaction(s) frauduleuse(s) detectee(s) sur un batch de "
            f"{n_total} transaction(s).\n"
            "Le detail complet des predictions est en piece jointe.\n\n"
            "-- ETL Fraud Detection (Airflow)"
        )
    else:
        subject = f"[RAS] Aucune fraude detectee sur {n_total} transaction(s)"
        body = (
            "Bonjour,\n\n"
            f"RAS : aucune transaction frauduleuse sur le batch de {n_total} "
            "transaction(s).\n"
            "Le detail des predictions est en piece jointe.\n\n"
            "-- ETL Fraud Detection (Airflow)"
        )

    _send_email(subject, body, attachment_path, attachment_name)
    print(f"[INFO] Mail envoye (n_fraud={n_fraud}, n_total={n_total}).")
