# Variables
VENV = venv
PYTHON = $(VENV)/bin/python
DATA = merged_data.csv
TARGET = Churn
MODEL = random_forest
MODEL_FILE = model.pkl
REQUIREMENTS = requirements.txt
MAIN_SCRIPT = model_pipeline.py

# Étape 1 : Installer les dépendances
install:
	python3 -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -r $(REQUIREMENTS)

# Étape 2 : Vérification du code
lint:
	. $(VENV)/bin/activate && flake8 

format:
	. $(VENV)/bin/activate && black .


security:
	. $(VENV)/bin/activate && bandit -r . --exit-zero

# Étape 3 : Préparer les données
prepare-data:
	$(PYTHON) $(MAIN_SCRIPT) --data $(DATA) --target $(TARGET) --action "prepare"

# Étape 4 : Entraîner le modèle
train:
	$(PYTHON) $(MAIN_SCRIPT) --data $(DATA) --target $(TARGET) --model $(MODEL_FILE) --action "train"

# Étape 5 : Évaluer le modèle
evaluate:
	$(PYTHON) $(MAIN_SCRIPT) --data $(DATA) --target $(TARGET) --model $(MODEL_FILE) --action "evaluate"
run-api:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Étape 7 : Exécution automatique
all: install lint format security prepare-data train test

# Commande pour exécuter la pipeline complète (en cas de modifications dans les fichiers)
run-all: install lint format security prepare-data train evaluate test
