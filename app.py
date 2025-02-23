from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import joblib  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore
import os
import logging


# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Définition de l'API
app = FastAPI(
    title="API de Prédiction de Churn",
    description="Cette API prédit si un client va résilier son abonnement.",
    version="1.3.1",
)


# Attachement du dossier `static` pour servir les fichiers HTML
app.mount("/static", StaticFiles(directory="static"), name="static")


# Chemin des modèles
MODEL_PATH = "model.pkl"
ENCODER_PATH = "encoder.pkl"


# Chargement du modèle de machine learning
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    logger.info("✅ Modèle chargé avec succès.")
else:
    logger.error("❌ Erreur : Modèle non trouvé !")
    model = None


# Chargement de l'encodeur
if os.path.exists(ENCODER_PATH):
    encoder = joblib.load(ENCODER_PATH)
    logger.info("✅ Encodeur chargé avec succès.")
else:
    logger.warning("⚠️ Encodeur non trouvé. Utilisation d'un LabelEncoder par défaut.")
    encoder = LabelEncoder()


# Liste des features attendues
FEATURE_NAMES = [
    "State",
    "Account length",
    "Area code",
    "International plan",
    "Voice mail plan",
    "Number vmail messages",
    "Total day minutes",
    "Total day calls",
    "Total day charge",
    "Total eve minutes",
    "Total eve calls",
    "Total eve charge",
    "Total night minutes",
    "Total night calls",
    "Total night charge",
    "Total intl minutes",
    "Total intl calls",
    "Total intl charge",
    "Customer service calls",
]


# Classe pour les requêtes de prédiction
class PredictionRequest(BaseModel):
    features: dict


# Classe pour les requêtes de réentraînement
class RetrainRequest(BaseModel):
    data: list[list[float]]
    labels: list[int]


# Route pour afficher l'interface HTML
@app.get("/", response_class=HTMLResponse, summary="Interface utilisateur")
async def serve_frontend():
    file_path = os.path.join("static", "index.html")
    if not os.path.exists(file_path):
        logger.error("❌ Fichier HTML introuvable !")
        raise HTTPException(status_code=404, detail="Fichier HTML non trouvé")

    with open(file_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# Route pour effectuer une prédiction
@app.post("/predict", summary="Effectue une prédiction de churn")
async def predict(request: PredictionRequest):
    logger.info("🔍 Début de la prédiction...")

    if model is None:
        logger.error("❌ Modèle non chargé !")
        raise HTTPException(status_code=500, detail="Modèle non chargé.")

    if not isinstance(request.features, dict):
        logger.error("❌ Les features doivent être un dictionnaire.")
        raise HTTPException(
            status_code=400, detail="Les features doivent être un dictionnaire."
        )

    missing_features = [f for f in FEATURE_NAMES if f not in request.features]
    if missing_features:
        logger.error(f"❌ Features manquantes : {missing_features}")
        raise HTTPException(
            status_code=400, detail=f"Features manquantes : {missing_features}"
        )

    logger.info(f"✅ Caractéristiques reçues : {request.features}")

    transformed_features = []
    for feature in FEATURE_NAMES:
        value = request.features[feature]
        logger.info(f"🛠️ Traitement de {feature} : {value}")
        if isinstance(value, str):
            try:
                encoded_value = encoder.transform([value])[0]
                transformed_features.append(encoded_value)
            except ValueError:
                logger.error(
                    f"❌ Valeur inconnue dans les features : {feature} ({value})"
                )
                raise HTTPException(
                    status_code=400, detail=f"Valeur inconnue : {feature} ({value})"
                )
        else:
            transformed_features.append(value)

    input_data = np.array(transformed_features).reshape(1, -1)

    try:
        prediction = model.predict(input_data).tolist()
        logger.info(f"✅ Prédiction réussie : {prediction}")
    except Exception as e:
        logger.error(f"❌ Erreur de prédiction : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")

    return {"prediction": prediction}


# Route pour réentraîner le modèle
@app.post("/retrain", summary="Réentraîne le modèle avec de nouvelles données")
def retrain(request: RetrainRequest):
    logger.info("🔄 Début du réentraînement du modèle...")

    if model is None:
        logger.error("❌ Modèle non chargé !")
        raise HTTPException(status_code=500, detail="Modèle non chargé.")

    X_train = np.array(request.data)
    y_train = np.array(request.labels)

    try:
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)
        logger.info("✅ Modèle réentraîné et sauvegardé avec succès.")
        return {"message": "Modèle réentraîné avec succès"}
    except Exception as e:
        logger.error(f"❌ Erreur lors du réentraînement : {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# Tout est ajusté avec 2 lignes vides là où c'était nécessaire
# Tu peux relancer `make lint` pour vérifier ! 🚀
