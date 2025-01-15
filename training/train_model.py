import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Caricamento del dataset
dataset_path = "../datasets/ready_to_use/selected_features_impatto_ambientale.csv"  # Modifica con il percorso corretto
data = pd.read_csv(dataset_path)

# Separazione tra feature (X) e target (y)
X = data.drop("PunteggioImpattoAmbientale", axis=1)  # Sostituisci con il nome esatto della tua colonna target
y = data["PunteggioImpattoAmbientale"]

# Suddivisione in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definizione del modello con i migliori iperparametri
model = XGBRegressor(
    learning_rate=0.1,
    max_depth=3,
    n_estimators=200,
    random_state=42,
    verbosity=2,
    booster='gbtree'  # Booster predefinito
)

# Ottimizzatori alternativi
booster_options = ['gbtree', 'gblinear', 'dart']

# Training con diversi ottimizzatori
best_rmse = float("inf")
best_booster = None
best_model = None

for booster in booster_options:
    print(f"\nTraining con booster: {booster}")
    model.set_params(booster=booster)
    model.fit(X_train, y_train)

    # Valutazione sul test set
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE con booster {booster}: {rmse:.4f}")

    # Salvare il miglior modello
    if rmse < best_rmse:
        best_rmse = rmse
        best_booster = booster
        best_model = model

# Salvataggio del miglior modello
model_save_path = "models/best_xgboost_model.pkl"
joblib.dump(best_model, model_save_path)
print(f"\nMiglior modello salvato in: {model_save_path}")
print(f"Miglior booster: {best_booster}, RMSE: {best_rmse:.4f}")

# Valutazione finale del miglior modello
y_final_pred = best_model.predict(X_test)
final_rmse = np.sqrt(mean_squared_error(y_test, y_final_pred))
print(f"\nValutazione finale sul test set con il miglior modello:")
print(f"RMSE: {final_rmse:.4f}")
