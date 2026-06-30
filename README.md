# Certification Architecte en Intelligence Artificielle - Bloc 3

**Candidat :** Mehdi Semar  
**Livrable :** Conception et mise en œuvre de pipelines de données (Workflow Orchestration)

---

## Contexte du Projet

Ce dépôt GitHub constitue le livrable technique pour le **Bloc 3 de la certification d'Architecte en Intelligence Artificielle**.
Il démontre la capacité à concevoir, industrialiser et orchestrer des pipelines de données robustes et sécurisés pour des projets d'Intelligence Artificielle.

### Attentes du Jury validées par ce projet :

- [x] **Concevoir les flux** (batch, streaming, ETL/ELT).
- [x] **Structurer les transformations** (nettoyage, feature engineering).
- [x] **Industrialiser et orchestrer** (via Apache Airflow).
- [x] **Mettre en place la qualité des données** (tests, gestion des anomalies).
- [x] **Intégrer la sécurité et la conformité** (gestion des secrets, RGPD).
- [x] **Superviser les performances et les coûts** (Monitoring).
- [x] **Assurer la traçabilité** (Data lineage, MLflow) et piloter en mode agile.

---

## Architecture du Projet

L'arborescence du projet est conçue de manière modulaire pour séparer l'orchestration, l'entraînement, le tracking et le déploiement.

```
BLOC3_WORKFLOW_ORCHESTRATION/
│
├── airflow_server/        # Environnement complet
│   │                        d'orchestration
│   ├── dags/                 # Pipelines de données
│   │                        (extraction, transformation,
│   │                        chargement)
│   ├── data/                 # Stockage local des données (Bronze, Silver, Gold)
│   ├── logs/                 # Traces d'exécution pour la supervision
│   ├── plugins/              # Opérateurs et hooks personnalisés Airflow
│   ├── docker-compose.yaml   # Orchestration des conteneurs Airflow (Webserver, Scheduler, etc.)
│   ├── Dockerfile            # Personnalisation de l'image de base Airflow
│   ├── README.md             # Documentation interne du serveur Airflow
│   └── requirements.txt      # Dépendances Python pour l'orchestration
│
├──  api/                   # API de serving pour le modèle de Machine Learning classique
├──  apiHF/                 # API de serving dédiée aux modèles Hugging Face
│
├──  mlflow/                # Serveur de tracking de performances et data lineage (ML classique)
├──  mlflowHF/              # Tracking et registre de modèles spécifique à Hugging Face
│
├──  train/                 # Scripts d'entraînement, nettoyage et feature engineering
│
├──  .env                   # Configuration locale et variables d'environnement (non versionné)
├──  .gitignore             # Exclusion des fichiers temporaires, données brutes et secrets
└──  secrets.sh             # Script d'initialisation et de chiffrement des variables sensibles
```
