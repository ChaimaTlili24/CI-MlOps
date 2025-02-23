# Variables
VENV = venv
PYTHON = $(VENV)/bin/python
DATA = merged_data.csv
TARGET = Churn
MODEL = random_forest
MODEL_FILE = model.pkl
REQUIREMENTS = requirements.txt
MAIN_SCRIPT = model_pipeline.py
NOTEBOOK = analysis_notebook.ipynb  # Nom de ton notebook Jupyter

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
	$(VENV)/bin/python $(MAIN_SCRIPT) --train --data $(DATA) --target $(TARGET) --model $(MODEL)

# Étape 5 : Évaluer le modèle
evaluate:
	$(PYTHON) $(MAIN_SCRIPT) --data $(DATA) --target $(TARGET) --model $(MODEL_FILE) --action "evaluate"

# Étape 6 : Tester le modèle
test:
	$(VENV)/bin/python -m unittest discover tests

# Étape 7 : Mesurer la performance
performance:
	$(VENV)/bin/python $(MAIN_SCRIPT) --data $(DATA) --target $(TARGET) --action "evaluate_performance"

# Étape 8 : Optimiser le modèle (par exemple, validation croisée)
optimize:
	$(VENV)/bin/python $(MAIN_SCRIPT) --data $(DATA) --target $(TARGET) --action "optimize"

# Étape 9 : Suivi de l'excellence
excellence:
	$(VENV)/bin/python $(MAIN_SCRIPT) --data $(DATA) --target $(TARGET) --action "log_excellence"

# Étape 10 : Démarrer Jupyter Notebook
jupyter:
	. $(VENV)/bin/activate && jupyter notebook --no-browser --port=8888

# Étape 11 : Exécuter un notebook spécifique
run-notebook:
	. $(VENV)/bin/activate && jupyter nbconvert --execute $(NOTEBOOK) --to notebook --inplace

# Commande pour exécuter la pipeline complète (en cas de modifications dans les fichiers)
all: install lint format security prepare-data train evaluate performance optimize excellence test

run-api:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000
