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
