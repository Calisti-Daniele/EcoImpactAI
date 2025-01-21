from utilities.functions import *
import matplotlib.pyplot as plt
import seaborn as sns

# Caricare il dataset
dataset_path = "../datasets/ready_to_use/preprocessed_impatto_ambientale.csv"
data = load_dataset(dataset_path)


# Informazioni generali sul dataset
print("Informazioni sul dataset:")
print(data.info())

print("\nPrime righe del dataset:")
print(data.head())

# Descrizione statistica
print("\nDescrizione statistica:")
print(data.describe())

# Controllare valori nulli
print("\nValori nulli per colonna:")
print(data.isnull().sum())

# Visualizzazione delle distribuzioni
plt.figure(figsize=(10, 6))
for column in data.select_dtypes(include=['float64', 'int64']).columns:
    sns.histplot(data[column], kde=True, bins=30)
    plt.title(f"Distribuzione di {column}")
    plt.show()

# Matrice di correlazione
plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matrice di Correlazione")
plt.show()

# Boxplot per identificare outlier
plt.figure(figsize=(10, 6))
for column in data.select_dtypes(include=['float64', 'int64']).columns:
    sns.boxplot(x=data[column])
    plt.title(f"Boxplot di {column}")
    plt.show()

# Distribuzione delle variabili categoriche
categorical_columns = data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(data[column])
    plt.title(f"Distribuzione di {column}")
    plt.xticks(rotation=45)
    plt.show()
