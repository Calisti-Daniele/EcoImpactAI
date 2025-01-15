import streamlit as st
import pandas as pd
import joblib
import time
import sys
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Aggiungi la directory principale del progetto al Python Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilities.functions import set_df_for_web_prediction

# Caricamento delle variabili d'ambiente
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

# Funzione per ottenere la spiegazione e formattarla
def get_explanation(punteggio, parametri):

    content = f"""
    Sei un consulente esperto in sostenibilità ambientale. Il punteggio di impatto ambientale calcolato per l'edificio è {punteggio}%.
    Parametri dell'edificio:
    {parametri}
    Analizza i parametri forniti e rispondi in modo chiaro e organizzato. Includi:
    1. Una breve analisi del punteggio.
    2. I fattori principali che influenzano il punteggio.
    3. Tre suggerimenti pratici per migliorare la sostenibilità ambientale.
    Usa un linguaggio professionale e inserisci emoji per rendere il testo più accattivante.
    """

    client = InferenceClient(api_key=HF_API_KEY)

    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=messages,
        max_tokens=1200,
        temperature=0.5,
        top_p=0.9
    )

    # Estrazione e formattazione della risposta
    raw_response = completion.choices[0].message["content"]
    return raw_response

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

parametri = f"""
- Tipo di Edificio: {tipo_edificio}
- Dimensione Area: {dimensione_area} m²
- Materiale: {materiale}
- Emissioni di CO2: {emissioni_co2} kg
- Consumo Acqua: {consumo_acqua} m³
"""

# Previsione
if st.sidebar.button("Calcola Impatto Ambientale"):
    with st.spinner("Calcolo in corso..."):
        time.sleep(1)  # Simula il caricamento
        input_data = set_df_for_web_prediction(input_data, "../training/scaler/scaler.pkl")
        prediction = best_model.predict(input_data)

        # Risultato
        st.write("### Risultato:")
        st.metric("Punteggio di Impatto Ambientale", f"{prediction[0]:.2f}%")

        # Spiegazione
        explanation = get_explanation(round(prediction[0]), parametri)

        # Mostra la spiegazione in una card
        st.markdown(
            f"""
            <div style="box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2); padding: 16px; border-radius: 8px; background-color: #f9f9f9;">
                <h3 style="color: #4CAF50;">Spiegazione del Risultato</h3>
                <p>{explanation}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
