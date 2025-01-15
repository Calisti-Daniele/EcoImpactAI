import pandas as pd

# Carico i due dataset
creditcard_data = pd.read_csv('../datasets/impatto_ambientale.csv')
fraud_creditcard_data = pd.read_csv('../datasets/synthetic_impatto_ambientale.csv')

# Controllo che abbiano le stesse colonne
print(f"Colonne creditcard.csv: {list(creditcard_data.columns)}")
print(f"Colonne fraud_creditcard.csv: {list(fraud_creditcard_data.columns)}")

# Unisco i dataset
merged_data = pd.concat([creditcard_data, fraud_creditcard_data], ignore_index=True)

# Rimuovo eventuali duplicati
merged_data = merged_data.drop_duplicates()

# Salvo il dataset unito
merged_data.to_csv('../datasets/merged_impatto_ambientale.csv', index=False)

# Verifico le dimensioni del dataset finale
print(f"Dimensioni del dataset unito: {merged_data.shape}")
