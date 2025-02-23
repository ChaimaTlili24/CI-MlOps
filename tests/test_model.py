import mlflow
import unittest
from model_pipeline import prepare_data, train_model
import os


class TestModelTraining(unittest.TestCase):

    def test_train_model(self):
        # Définir l'URI de suivi de MLflow
        mlflow.set_tracking_uri(
            "http://localhost:5003"
        )  # Assurez-vous que le port est correct pour ton serveur MLflow

        # Préparer les données
        X_train_res, X_test, y_train_res, y_test = prepare_data(
            "/home/chaima/ml_project_chaimatlili/merged_data.csv", target_column="Churn"
        )

        # Spécifier le chemin du modèle
        model_path = "model.pkl"

        # Exécuter la fonction d'entraînement
        train_model(X_train_res, y_train_res, X_test, y_test, model_path)

        # Vérifier si le modèle a été correctement enregistré
        self.assertTrue(
            os.path.exists(model_path)
        )  # Vérifie que le fichier du modèle existe

        # Vérification des métriques loggées dans MLflow (Exemple d'assertion pour vérifier si la métrique existe)
        with mlflow.start_run():
            # Récupérer les métriques de la dernière exécution
            metrics = mlflow.search_runs(order_by=["start_time desc"]).iloc[0]

            # Afficher toutes les métriques pour mieux comprendre ce qui est loggé
            print("Métriques loggées dans MLflow : ", metrics.to_dict())


if __name__ == "__main__":
    unittest.main()
