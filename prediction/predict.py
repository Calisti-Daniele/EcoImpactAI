from utilities.functions import *
import joblib

# Caricamento del modello salvato
model_path = "../training/models/best_xgboost_model.pkl"
best_model = joblib.load(model_path)
print(f"Modello caricato da: {model_path}")

# Caricamento dei dati non visti
new_data_path = "../datasets/for_prediction/for_prediction.csv"  # Sostituisci con il percorso corretto
new_data = load_dataset(new_data_path)

new_data = encode_categorical(new_data)

# Effettuare le previsioni
predictions = best_model.predict(new_data)

# Aggiungere le previsioni al DataFrame
new_data["Previsione_ImpattoAmbientale"] = predictions

# Salvataggio del risultato
output_path = "../datasets/for_prediction/previsioni_dati_non_visti.csv"
new_data.to_csv(output_path, index=False)
print(f"Previsioni salvate in: {output_path}")
