from utilities.functions import *

df = load_dataset("../datasets/merged_impatto_ambientale.csv")

#Creo nuove feature da quelle che ho già
df = create_new_features(df)

#Faccio l'encoding delle feature "categorical"
df = encode_categorical(df)

df.to_csv("../datasets/ready_to_use/preprocessed_impatto_ambientale.csv")