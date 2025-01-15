import streamlit as st
import pandas as pd
import joblib

import sys
import os

# Aggiungi la directory principale del progetto al Python Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilities.functions import set_df_for_web_prediction

# Caricamento del modello salvato
model_path = "../training/models/best_xgboost_model.pkl"
best_model = joblib.load(model_path)

# Titolo della dashboard
st.title("EcoImpactAI 🌍")
st.subheader("Prevedi l'impatto ambientale delle costruzioni con l'AI")

# Input dei parametri
st.sidebar.header("Inserisci i dati dell'edificio")
tipo_edificio = st.sidebar.selectbox("Tipo di Edificio", ["Residenziale", "Commerciale", "Industriale"])
dimensione_area = st.sidebar.number_input("Dimensione Area (m²)", min_value=50, max_value=1000, step=1)
materiale = st.sidebar.selectbox("Materiale", ["Cemento", "Legno", "Acciaio", "Misto"])
efficienza_energetica = st.sidebar.slider("Efficienza Energetica", min_value=0.5, max_value=1.0, step=0.01)
emissioni_co2 = st.sidebar.number_input("Emissioni CO2 (kg)", min_value=100, max_value=10000, step=100)
impatto_biodiversita = st.sidebar.number_input("Impatto Biodiversità (%)", min_value=0, max_value=20, step=1)
prossimita_natura = st.sidebar.number_input("Prossimità Natura (km)", min_value=0, max_value=50, step=1)
consumo_acqua = st.sidebar.number_input("Consumo Acqua (m³)", min_value=100, max_value=5000, step=100)

# Mappature (encoding)
encoding_maps = {
    "TipoEdificio": {"Residenziale": 0, "Commerciale": 1, "Industriale": 2},
    "Materiale": {"Cemento": 0, "Legno": 1, "Acciaio": 2, "Misto": 3},
}

# Preparazione dei dati
input_data = pd.DataFrame({
    "EfficienzaEnergetica": [efficienza_energetica],
    "EmissioniCO2": [emissioni_co2],
    "ImpattoBiodiversita": [impatto_biodiversita],
    "ConsumoAcqua": [consumo_acqua],
    "EfficienzaInversa": None
})

# Previsione
if st.sidebar.button("Calcola Impatto Ambientale"):
    input_data = set_df_for_web_prediction(input_data, "../training/scaler/scaler.pkl")

    prediction = best_model.predict(input_data)
    st.write("### Risultato:")

    print(prediction)
    st.metric("Punteggio di Impatto Ambientale", f"{prediction[0]:.2f}%")
