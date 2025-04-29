import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os

# Paths
DESCRIPTORS_PATH = "data/processed/descriptors.csv"
MODEL_PATH = "data/processed/model.pkl"
PLOT_PATH_SCATTER = "data/processed/pred_vs_actual.png"
PLOT_PATH_IMPORTANCE = "data/processed/feature_importance.png"

def create_visualizations():
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
    
    # Create scatter plot: Predicted vs Actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, alpha=0.5, color='blue')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Activity')
    plt.ylabel('Predicted Activity')
    plt.title('Predicted vs Actual Activity')
    plt.grid(True)
    os.makedirs(os.path.dirname(PLOT_PATH_SCATTER), exist_ok=True)
    plt.savefig(PLOT_PATH_SCATTER)
    plt.close()
    print(f"Scatter plot saved to {PLOT_PATH_SCATTER}")
    
    # Create feature importance plot
    feature_names = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors']
    importances = model.feature_importances_
    
    plt.figure(figsize=(8, 6))
    plt.bar(feature_names, importances, color='green')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance in Random Forest Model')
    plt.grid(True, axis='y')
    os.makedirs(os.path.dirname(PLOT_PATH_IMPORTANCE), exist_ok=True)
    plt.savefig(PLOT_PATH_IMPORTANCE)
    plt.close()
    print(f"Feature importance plot saved to {PLOT_PATH_IMPORTANCE}")

if __name__ == "__main__":
    create_visualizations()