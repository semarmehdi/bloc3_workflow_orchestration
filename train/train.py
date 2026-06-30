import argparse
import pandas as pd
import time
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration du tracking URI MLflow (Ex: S3 / HuggingFace / Local)
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

if __name__ == "__main__":

    ### Configuration de l'expérience MLflow
    experiment_name = "fraud_detection_experiment"
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    # Parse des arguments de la ligne de commande
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="random_forest",
        choices=["random_forest", "xgboost"],
        help="Modèle à entraîner",
    )
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=6)
    args = parser.parse_args()

    print(f"Démarrage de l'entraînement : {args.model_type}...")
    start_time = time.time()

    # 1. Import du dataset de fraude
    url = "https://lead-program-assets.s3.eu-west-3.amazonaws.com/M05-Projects/fraudTest.csv"
    df = pd.read_csv(url, index_col=0)

    # 2. Feature Engineering & Nettoyage de base
    # Extraction de composants temporels
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek

    # Calcul de l'âge au moment de la transaction
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365

    # Target & Features
    target_col = "is_fraud"
    y = df[target_col]

    # On supprime les colonnes ID, textuelles brutes ou trop cardinales (ex: street, trans_num)
    cols_to_drop = [
        target_col,
        "trans_date_trans_time",
        "cc_num",
        "first",
        "last",
        "street",
        "city",
        "trans_num",
        "dob",
        "unix_time",
    ]
    X = df.drop(columns=cols_to_drop)

    # 3. Split Train / Test (crucial pour valider sur de la fraude)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Définition des Transformers
    categorical_features = ["merchant", "category", "gender", "state", "job"]
    numerical_features = [
        "amt",
        "lat",
        "long",
        "city_pop",
        "merch_lat",
        "merch_long",
        "hour",
        "day_of_week",
        "age",
    ]

    categorical_transformer = OneHotEncoder(
        drop="first", handle_unknown="ignore", sparse_output=False
    )
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numerical_transformer, numerical_features),
        ]
    )

    # 5. Choix du classifieur
    if args.model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
            n_jobs=-1,
        )
    else:
        classifier = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
            n_jobs=-1,
        )

    # 6. Pipeline Imblearn (Prepro -> SMOTE -> Classifier)
    model_pipeline = ImbPipeline(
        steps=[
            ("Preprocessing", preprocessor),
            ("SMOTE", SMOTE(random_state=42)),
            ("Classifier", classifier),
        ],
        verbose=True,
    )

    # 7. Log de l'expérience dans MLflow
    with mlflow.start_run() as run:
        # Log des hyperparamètres manuellement (vu qu'on a coupé l'autolog)
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        # Entraînement
        model_pipeline.fit(X_train, y_train)

        # Prédictions pour la signature
        predictions = model_pipeline.predict(X_test)

        # Métriques
        mlflow.log_metric("precision", precision_score(y_test, predictions))
        mlflow.log_metric("recall", recall_score(y_test, predictions))
        mlflow.log_metric("f1_score", f1_score(y_test, predictions))

        # rapport complet sous forme de texte
        report = classification_report(y_test, predictions)
        mlflow.log_text(report, "classification_report.txt")
        # ------------------------------------------------

        # Signature et exemple d'entrée
        signature = infer_signature(X_test, predictions)
        input_example = X_test.head(3)
        registered_model_name = "fraud_detection"

        # Enregistrement du modèle (S3 géré automatiquement via MLflow si configuré en backend)
        if args.model_type == "random_forest":
            model_info = mlflow.sklearn.log_model(
                sk_model=model_pipeline,
                name="model",
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example,
            )
        else:
            model_info = mlflow.sklearn.log_model(
                sk_model=model_pipeline,
                name="model",
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example,
            )

        # Gestion de l'alias "challenger"
        alias_name = "challenger"
        model_version = model_info.registered_model_version
        print(
            f"[INFO] Modèle ({args.model_type}) enregistré sous la version {model_version}"
        )

        client.set_registered_model_alias(
            name=registered_model_name,
            alias=alias_name,
            version=model_version,
        )

        print(
            f"[INFO] L'alias '{alias_name}' pointe désormais sur la version {model_version}"
        )

    print("...Terminé !")
    print(f"--- Temps total d'exécution : {time.time() - start_time} secondes")
