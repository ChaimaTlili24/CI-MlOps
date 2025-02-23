import argparse
import mlflow
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    load_model
)

# Configuration de l'argument parser
parser = argparse.ArgumentParser(description="Pipeline de classification Machine Learning avec MLflow")
parser.add_argument("--data", required=True, help="Chemin du fichier CSV, ex: /home/chaima/ml_project_chaimatlili/merged_data.csv")
parser.add_argument("--target", required=True, help="Nom de la colonne cible, ex: Churn")
parser.add_argument("--model", required=True, help="Chemin pour sauvegarder/charger le modèle, ex: model.pkl")
parser.add_argument("--action", required=True, choices=["train", "evaluate"], help="Action à effectuer (train/evaluate)")
parser.add_argument("--test_size", type=float, default=0.2, help="Proportion des données de test (par défaut: 0.2)")
parser.add_argument("--random_state", type=int, default=42, help="Graine aléatoire pour la reproductibilité")

args = parser.parse_args()

# Activation du tracking MLflow
mlflow.set_tracking_uri("http://0.0.0.0:5002")
mlflow.set_experiment("RandomForest_Classification")
mlflow.set_tracking_uri("http://localhost:5003")

# Créer ou accéder à une expérience
mlflow.set_experiment("my_experiment")
with mlflow.start_run(run_name="test_run"):
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    print("Expérience loggée !")
def main():
    if args.action == "train":
        print("### Préparation des données...")
        X_train_res, X_test, y_train_res, y_test = prepare_data(
            filepath=args.data, 
            target_column=args.target, 
            test_size=args.test_size, 
            random_state=args.random_state
        )

        print("### Entraînement du modèle et log MLflow...")
        train_model(
            X_train=X_train_res, 
            y_train=y_train_res, 
            X_test=X_test, 
            y_test=y_test, 
            model_path=args.model, 
            random_state=args.random_state
        )

    elif args.action == "evaluate":
        print("### Préparation des données...")
        _, X_test, _, y_test = prepare_data(
            filepath=args.data, 
            target_column=args.target, 
            test_size=args.test_size, 
            random_state=args.random_state
        )

        print("### Évaluation du modèle et log MLflow...")
        evaluate_model(model_path=args.model, X_test=X_test, y_test=y_test)

if __name__ == "__main__":
    main()
