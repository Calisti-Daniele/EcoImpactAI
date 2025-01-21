from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Funzione per calcolare RMSE
def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Caricamento del dataset
dataset_path = "../datasets/ready_to_use/selected_features_impatto_ambientale.csv"  # Modifica con il percorso corretto
data = pd.read_csv(dataset_path)

# Separazione tra feature (X) e target (y)
X = data.drop("PunteggioImpattoAmbientale", axis=1)  # Sostituisci con il nome esatto della tua colonna target
y = data["PunteggioImpattoAmbientale"]

# Suddivisione in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelli e iperparametri
models_and_parameters = {
    "RandomForest": {
        "model": RandomForestRegressor(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
    },
    "XGBoost": {
        "model": XGBRegressor(random_state=42, verbosity=2),
        "params": {
            "learning_rate": [0.01, 0.1, 0.2],
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 10]
        }
    }
}

# Scorer per GridSearchCV
scorer = make_scorer(rmse_scorer, greater_is_better=False)

# Esecuzione della Grid Search con raccolta dei risultati
best_models = {}
cv_results = {}  # Per memorizzare i risultati della validazione incrociata
for model_name, config in models_and_parameters.items():
    print(f"Grid Search per: {model_name}")
    grid_search = GridSearchCV(
        estimator=config["model"],
        param_grid=config["params"],
        scoring=scorer,
        cv=5,
        verbose=3,  # Abilita output dettagliato
        n_jobs=-1,
        return_train_score=True  # Include i punteggi di training
    )
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator_
    cv_results[model_name] = grid_search.cv_results_
    print(f"Miglior modello per {model_name}: {grid_search.best_params_}")
    print(f"RMSE: {abs(grid_search.best_score_):.4f}")

# Creazione del grafico
plt.figure(figsize=(12, 6))
for model_name, results in cv_results.items():
    mean_test_rmse = -results["mean_test_score"]  # Converte RMSE negativo in positivo
    params = range(1, len(mean_test_rmse) + 1)  # Genera una sequenza per ogni iperparametro testato
    plt.plot(params, mean_test_rmse, marker='o', label=model_name)

plt.title("Prestazioni in termini di RMSE durante la validazione incrociata", fontsize=14)
plt.xlabel("Configurazioni di iperparametri", fontsize=12)
plt.ylabel("RMSE medio", fontsize=12)
plt.legend(title="Modelli")
plt.grid()
plt.tight_layout()
plt.show()


# Selezione del miglior modello globale
best_model_name = min(best_models, key=lambda name: abs(grid_search.best_score_))
print(f"Miglior modello globale: {best_model_name}")

# Valutazione sul test set
best_model = best_models[best_model_name]
y_pred = best_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE sul test set per {best_model_name}: {test_rmse:.4f}")
