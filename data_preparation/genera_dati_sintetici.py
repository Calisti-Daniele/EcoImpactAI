import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata


# Carica il dataset reale
data = pd.read_csv('../datasets/impatto_ambientale.csv')

# Visualizza le prime righe del dataset
print(data.head())

# Crea la metadata del dataset
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)

# Inizializza il sintetizzatore con la metadata
synthesizer = GaussianCopulaSynthesizer(metadata)

# Addestra il sintetizzatore
synthesizer.fit(data)

# Genera dati sintetici
synthetic_data = synthesizer.sample(num_rows=5000)

# Visualizza le prime righe dei dati sintetici
print(synthetic_data.head())

# Salva i dati sintetici
synthetic_data.to_csv('../datasets/synthetic_impatto_ambientale.csv', index=False)