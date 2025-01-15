from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import joblib
from utilities.functions import *
from sklearn.model_selection import train_test_split

# Caricamento del modello salvato
model_path = "../training/models/best_xgboost_model.pkl"  # Percorso del modello salvato
best_model = joblib.load(model_path)
print(f"Modello caricato da: {model_path}")

# Caricamento del dataset
dataset_path = "../datasets/ready_to_use/selected_features_impatto_ambientale.csv"  # Modifica con il percorso corretto
data = load_dataset(dataset_path)

# Separazione tra feature (X) e target (y)
X = data.drop("PunteggioImpattoAmbientale", axis=1)  # Sostituisci con il nome esatto della tua colonna target
y = data["PunteggioImpattoAmbientale"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calcolo delle metriche sul test set
y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Valutazione sul test set:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X, y, cv=kf, scoring='neg_root_mean_squared_error')

print("\nRisultati della Cross-validation:")
print(f"RMSE medio (CV): {-np.mean(cv_scores):.4f}")
print(f"Deviazione standard (CV): {np.std(cv_scores):.4f}")

# Grafico: Confronto tra valori reali e previsti
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([0, 100], [0, 100], '--r', label="Previsione Perfetta")
plt.xlabel("Valori Reali")
plt.ylabel("Valori Previsti")
plt.title("Confronto Valori Reali vs Previsti")
plt.legend()
plt.grid()
plt.show()

# Grafico: Residui
residui = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.hist(residui, bins=30, alpha=0.7)
plt.axvline(x=0, color='red', linestyle='--')
plt.xlabel("Errore (Residui)")
plt.ylabel("Frequenza")
plt.title("Distribuzione degli Errori")
plt.grid()
plt.show()
