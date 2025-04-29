import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import os

# Paths
PROCESSED_DATA_PATH = "data/processed/processed_dataset.csv"
DESCRIPTORS_PATH = "data/processed/descriptors.csv"

def compute_descriptors():
    # Read processed data
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # Initialize descriptor lists
    descriptors = []
    
    for smiles in df['SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            desc = {
                'MolWt': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'NumHDonors': Descriptors.NumHDonors(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol)
            }
        else:
            desc = {'MolWt': None, 'LogP': None, 'NumHDonors': None, 'NumHAcceptors': None}
        descriptors.append(desc)
    
    # Create descriptors DataFrame
    desc_df = pd.DataFrame(descriptors)
    result_df = pd.concat([df[['SMILES', 'Activity']], desc_df], axis=1)
    
    # Save descriptors
    os.makedirs(os.path.dirname(DESCRIPTORS_PATH), exist_ok=True)
    result_df.to_csv(DESCRIPTORS_PATH, index=False)
    print(f"Descriptors saved to {DESCRIPTORS_PATH}")

if __name__ == "__main__":
    compute_descriptors()