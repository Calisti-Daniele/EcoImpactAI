from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib
import pandas as pd

# Caricamento del modello salvato
model_path = "../training/models/best_xgboost_model.pkl"  # Percorso del modello salvato
best_model = joblib.load(model_path)
print(f"Modello caricato da: {model_path}")

# Caricamento del dataset
dataset_path = "../datasets/ready_to_use/selected_features_impatto_ambientale.csv"  # Modifica con il percorso corretto
data = pd.read_csv(dataset_path)

# Separazione tra feature (X) e target (y)
X = data.drop("PunteggioImpattoAmbientale", axis=1)  # Sostituisci con il nome esatto della tua colonna target
y = data["PunteggioImpattoAmbientale"]

# Configurazione della cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Inizializzare liste per raccogliere i risultati
mae_scores = []
rmse_scores = []
r2_scores = []

# Esecuzione della k-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Addestramento del modello
    best_model.fit(X_train, y_train)

    # Previsioni
    y_pred = best_model.predict(X_test)

    # Calcolo delle metriche
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Salvare i risultati
    mae_scores.append(mae)
    rmse_scores.append(rmse)
    r2_scores.append(r2)

# Stampa dei risultati medi
print("\nRisultati della Cross-validation:")
print(f"MAE medio (CV): {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"RMSE medio (CV): {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"R² medio (CV): {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
