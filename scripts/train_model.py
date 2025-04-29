import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

# Paths
DESCRIPTORS_PATH = "data/processed/descriptors.csv"
MODEL_PATH = "data/processed/model.pkl"

def train_model():
    # Read descriptors
    df = pd.read_csv(DESCRIPTORS_PATH)
    
    # Prepare features and target
    X = df[['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors']].dropna()
    y = df.loc[X.index, 'Activity']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()