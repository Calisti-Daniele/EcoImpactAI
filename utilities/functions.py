import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_dataset(path):

    df = pd.read_csv(path)

    return df

def encode_categorical(df):

    encoding_maps = {
        "TipoEdificio": {"Residenziale": 0, "Commerciale": 1, "Industriale": 2},
        "Materiale": {"Cemento": 0, "Legno": 1, "Acciaio": 2, "Misto": 3},
        "ConformitaNormativa": {"Si": 1, "No": 0},
    }

    # Applicare il mapping a ciascuna colonna del dizionario
    for column, mapping in encoding_maps.items():
        if column in df.columns:  # Controlla che la colonna esista nel dataframe
            df[column] = df[column].map(mapping)

    return df

def normalizza_feature_numeriche(df):
    # Normalizzazione delle feature numeriche
    scaler = MinMaxScaler()
    numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns.drop("PunteggioImpattoAmbientale")
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    return df


def create_new_features(df):
    # Creazione di nuove feature
    # 1. Rapporto tra dimensione dell'area e consumo d'acqua
    df["Area_Acqua_Ratio"] = df["DimensioneArea"] / (df["ConsumoAcqua"] + 1e-6)

    # 2. Prossimità inversa alla natura (maggiore è la distanza, minore è la vicinanza)
    df["ProssimitaInversa"] = 1 / (df["ProssimitaNatura"] + 1e-6)

    # 3. Efficienza energetica normalizzata inversa
    df["EfficienzaInversa"] = 1 - df["EfficienzaEnergetica"]

    return df
