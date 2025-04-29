import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import AllChem

# Paths
RAW_DATA_PATH = "data/raw/dataset.csv"
PROCESSED_DATA_PATH = "data/processed/processed_dataset.csv"

def standardize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except:
        return None

def preprocess_data():
    # Read raw data
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Standardize SMILES
    df['SMILES'] = df['SMILES'].apply(standardize_smiles)
    
    # Remove invalid SMILES
    df = df.dropna(subset=['SMILES'])
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['SMILES'])
    
    # Save processed data
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    preprocess_data()