# QSAR Analysis Repository

This repository contains scripts and utilities for performing Quantitative Structure-Activity Relationship (QSAR) analysis using Python and Bash. It includes data preprocessing, feature extraction, model training, evaluation, and visualizations for predicting molecular activities based on chemical descriptors.

## Repository Structure

```
qsar-analysis/
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   ├── preprocess.py
│   ├── descriptors.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── visualize.py
├── run_workflow.sh
├── requirements.txt
├── README.md
└── LICENSE
```

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AhmedFikry90/QSAR_workflow_analysis.git
   cd qsar-analysis
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Data**:
   - Place your raw dataset (e.g., SMILES strings and activity values in CSV format) in the `data/raw/` directory.
   - Example dataset format:
   ```
   SMILES,Activity
   C1=CC=CC=C1,5.2
   ...
   ```

5. **Run the Workflow**:
   ```bash
   bash run_workflow.sh
   ```

## Scripts Overview

- **preprocess.py**: Cleans and preprocesses raw data, handling missing values and standardizing SMILES.
- **descriptors.py**: Computes molecular descriptors using RDKit (e.g., molecular weight, logP).
- **train_model.py**: Trains a Random Forest model on the processed data.
- **evaluate.py**: Evaluates the model performance using metrics like RMSE and R².
- **visualize.py**: Generates visualizations, including a scatter plot of predicted vs. actual activities and a feature importance plot.
- **run_workflow.sh**: Orchestrates the entire pipeline (preprocessing, descriptor calculation, training, evaluation, and visualization).

## Dependencies

Listed in `requirements.txt`:
```
pandas
numpy
rdkit
scikit-learn
matplotlib
```

## Example Usage

1. Ensure your dataset is in `data/raw/` (e.g., `dataset.csv`).
2. Run the workflow:
   ```bash
   bash run_workflow.sh
   ```
3. Outputs (processed data, descriptors, model, evaluation metrics, and visualizations) will be saved in `data/processed/`.

## License

MIT License. See `LICENSE` file for details.

---

Below are the contents of the key files in the repository, including the new `visualize.py` script and updated `run_workflow.sh` and `requirements.txt`:

### `run_workflow.sh`
```bash
#!/bin/bash

# QSAR Analysis Workflow
echo "Starting QSAR Analysis Workflow..."

# Activate virtual environment
source venv/bin/activate

# Run preprocessing
echo "Preprocessing data..."
python scripts/preprocess.py

# Compute descriptors
echo "Computing molecular descriptors..."
python scripts/descriptors.py

# Train model
echo "Training model..."
python scripts/train_model.py

# Evaluate model
echo "Evaluating model..."
python scripts/evaluate.py

# Generate visualizations
echo "Generating visualizations..."
python scripts/visualize.py

echo "Workflow completed!"
```

### `scripts/preprocess.py`
```python
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
```

### `scripts/descriptors.py`
```python
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
```

### `scripts/train_model.py`
```python
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
```

### `scripts/evaluate.py`
```python
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
```

### `scripts/visualize.py`
```python
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
```

### `requirements.txt`
```
pandas
numpy
rdkit
scikit-learn
matplotlib
```

### `LICENSE`
```
MIT License

Copyright (c) 2025 [AhmedFikry90]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```