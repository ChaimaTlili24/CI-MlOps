import numpy as np
import pandas as pd
import sklearn  # type: ignore
import mlflow

# Exemple d'utilisation pour éviter l'erreur F401
print("Using sklearn version:", sklearn.__version__)
print("Using mlflow:", mlflow.__version__)

print("Environment is set up correctly!")
