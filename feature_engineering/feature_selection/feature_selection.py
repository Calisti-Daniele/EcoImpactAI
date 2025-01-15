from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd

from utilities.functions import normalizza_feature_numeriche

# Carica il dataset
data = pd.read_csv('../../datasets/ready_to_use/preprocessed_impatto_ambientale.csv')

# Separazione tra feature e target
X = data.drop('PunteggioImpattoAmbientale', axis=1)  # Rimuovo la colonna target
y = data['PunteggioImpattoAmbientale']  # Target

# Applica SelectKBest con la funzione f_classif
selector = SelectKBest(score_func=f_regression, k=5)  # Seleziona le 10 migliori feature
X_new = selector.fit_transform(X, y)

# Ottieni i nomi delle feature selezionate
selected_features = X.columns[selector.get_support()]

# Visualizza le feature selezionate
print("Feature selezionate:")
print(selected_features)

# Salva il nuovo dataset con le feature selezionate
selected_data = pd.DataFrame(X_new, columns=selected_features)
selected_data['PunteggioImpattoAmbientale'] = y  # Riaggiungo la colonna target

#Prima normalizzo le feature numeriche
selected_data = normalizza_feature_numeriche(selected_data, "../../training/scaler/scaler.pkl")

selected_data.to_csv('../../datasets/ready_to_use/selected_features_impatto_ambientale.csv', index=False)
