import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Caricamento del modello salvato
model_path = "../training/models/best_xgboost_model.pkl"  # Modifica con il percorso corretto
best_model = joblib.load(model_path)
print(f"Modello caricato da: {model_path}")

# Caricamento del dataset
dataset_path = "../datasets/ready_to_use/selected_features_impatto_ambientale.csv"  # Modifica con il percorso corretto
data = pd.read_csv(dataset_path)

# Separazione tra feature (X) e target (y)
X = data.drop("PunteggioImpattoAmbientale", axis=1)  # Sostituisci con il nome esatto della tua colonna target
y = data["PunteggioImpattoAmbientale"]

# Suddivisione in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Previsioni
y_pred = best_model.predict(X_test)

# Calcolo dei residui (differenze tra valori reali e previsti)
residui = y_test - y_pred

# Grafico 1: Distribuzione dei residui
plt.figure(figsize=(10, 6))
plt.hist(residui, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='red', linestyle='--', label='Media residui = 0')
plt.title("Distribuzione dei residui", fontsize=16)
plt.xlabel("Errore (Residui)", fontsize=14)
plt.ylabel("Frequenza", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Grafico 2: Confronto tra valori previsti e reali
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='green', edgecolor='black', label="Previsioni")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label="Perfetta correlazione")
plt.title("Confronto tra valori previsti e reali", fontsize=16)
plt.xlabel("Valori reali", fontsize=14)
plt.ylabel("Valori previsti", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
