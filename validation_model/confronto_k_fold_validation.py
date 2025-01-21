from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Caricamento dei modelli
xgboost_model_path = "../training/models/best_xgboost_model.pkl"  # Percorso del modello XGBoost salvato
best_xgboost_model = joblib.load(xgboost_model_path)
print(f"Modello XGBoost caricato da: {xgboost_model_path}")

# Inizializzazione del modello Random Forest
random_forest_model = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=10)

# Caricamento del dataset
dataset_path = "../datasets/ready_to_use/selected_features_impatto_ambientale.csv"  # Modifica con il percorso corretto
data = pd.read_csv(dataset_path)

# Separazione tra feature (X) e target (y)
X = data.drop("PunteggioImpattoAmbientale", axis=1)  # Sostituisci con il nome esatto della tua colonna target
y = data["PunteggioImpattoAmbientale"]

# Configurazione della cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Inizializzare un dizionario per raccogliere i risultati
results = {
    "XGBoost": {"mae": [], "rmse": [], "r2": []},
    "RandomForest": {"mae": [], "rmse": [], "r2": []}
}

# Funzione per eseguire la K-Fold Validation
def perform_kfold_validation(model, model_name):
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Addestramento del modello
        model.fit(X_train, y_train)

        # Previsioni
        y_pred = model.predict(X_test)

        # Calcolo delle metriche
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Salvare i risultati
        results[model_name]["mae"].append(mae)
        results[model_name]["rmse"].append(rmse)
        results[model_name]["r2"].append(r2)

# Esecuzione della validazione per entrambi i modelli
perform_kfold_validation(best_xgboost_model, "XGBoost")
perform_kfold_validation(random_forest_model, "RandomForest")

# Calcolo delle medie e deviazioni standard
for model_name in results:
    print(f"\nRisultati della Cross-validation per {model_name}:")
    print(f"MAE medio: {np.mean(results[model_name]['mae']):.4f} ± {np.std(results[model_name]['mae']):.4f}")
    print(f"RMSE medio: {np.mean(results[model_name]['rmse']):.4f} ± {np.std(results[model_name]['rmse']):.4f}")
    print(f"R² medio: {np.mean(results[model_name]['r2']):.4f} ± {np.std(results[model_name]['r2']):.4f}")

# Creazione del grafico
models = ["XGBoost", "RandomForest"]
metrics = ["mae", "rmse", "r2"]
metric_names = ["MAE", "RMSE", "R²"]
colors = ["blue", "green"]

plt.figure(figsize=(10, 6))
for i, metric in enumerate(metrics):
    plt.subplot(1, 3, i + 1)  # Un grafico per ogni metrica
    for model, color in zip(models, colors):
        plt.bar(
            model,
            np.mean(results[model][metric]),
            yerr=np.std(results[model][metric]),
            label=f"{model} ({metric_names[i]})",
            color=color,
            alpha=0.7
        )
    plt.title(f"{metric_names[i]}")
    plt.ylabel(metric_names[i])
    plt.xticks(rotation=45)

plt.suptitle("Confronto delle metriche tra XGBoost e Random Forest", fontsize=16)
plt.tight_layout()
plt.show()
