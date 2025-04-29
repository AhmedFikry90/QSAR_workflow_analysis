import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import os

# Paths
DESCRIPTORS_PATH = "data/processed/descriptors.csv"
MODEL_PATH = "data/processed/model.pkl"
EVALUATION_PATH = "data/processed/evaluation.txt"

def evaluate_model():
    # Read descriptors
    df = pd.read_csv(DESCRIPTORS_PATH)
    
    # Prepare features and target
    X = df[['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors']].dropna()
    y = df.loc[X.index, 'Activity']
    
    # Load model
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    # Predict
    y_pred = model.predict(X)
    
    # Calculate metrics
    rmse = mean_squared_error(y, y_pred, squared=False)
    r2 = r2_score(y, y_pred)
    
    # Save evaluation
    os.makedirs(os.path.dirname(EVALUATION_PATH), exist_ok=True)
    with open(EVALUATION_PATH, 'w') as f:
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R² Score: {r2:.4f}\n")
    print(f"Evaluation metrics saved to {EVALUATION_PATH}")

if __name__ == "__main__":
    evaluate_model()