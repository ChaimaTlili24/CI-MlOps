import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from mlflow.models.signature import infer_signature

# ------------------------------
# 1) Chargement et préparation des données
# ------------------------------
def prepare_data(filepath, target_column, test_size=0.2, random_state=42):
    data = pd.read_csv(filepath)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    if y.dtypes == "object":
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    return X_train_res, X_test, y_train_res, y_test

# ------------------------------
# 2) Entraînement du modèle
# ------------------------------
def train_model(X_train, y_train, X_test, y_test, model_path, random_state=42):
    # Définir l'expérience dans MLflow
    mlflow.set_experiment("mlflow_project_experiment")

    # Commencer une exécution dans l'expérience définie
    with mlflow.start_run(run_name="model_training") as run:
        model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight="balanced")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        # Log des métriques
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1-score", f1)

        # Enregistrement du modèle
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        # Log du modèle complet avec signature
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, artifact_path="random_forest_model", signature=signature)

        print(f"Model trained and logged successfully! Accuracy: {accuracy:.4f}")

# ------------------------------
# 3) Évaluation du modèle
# ------------------------------
def evaluate_model(model_path, X_test, y_test):
    with mlflow.start_run(run_name="model_evaluation") as run:
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Log des métriques
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1-score", f1)

        # Log du rapport de classification
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        # Générer, afficher et sauvegarder la matrice de confusion
        plot_confusion_matrix(conf_matrix)
        mlflow.log_artifact("confusion_matrix.png")

        print(f"Model evaluated. Accuracy: {accuracy:.4f}")

# ------------------------------
# 4) Sauvegarde du modèle
# ------------------------------
def save_model(model, filepath):
    """
    Sauvegarde le modèle sur disque (pickle).
    """
    joblib.dump(model, filepath)

def load_model(filepath):
    """
    Charge un modèle depuis un fichier .pkl.
    """
    return joblib.load(filepath)

# ------------------------------
# 5) Génération de la matrice de confusion
# ------------------------------
def plot_confusion_matrix(conf_matrix, labels, title="Matrice de Confusion"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(title, fontsize=16)
    plt.xlabel("Predictions", fontsize=14)
    plt.ylabel("Valeurs Réelles", fontsize=14)
    image_path = "confusion_matrix.png"
    plt.savefig(image_path)
    plt.close()
    mlflow.log_artifact(image_path)