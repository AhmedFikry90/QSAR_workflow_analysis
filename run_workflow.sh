#!/bin/bash

# QSAR Analysis Workflow
echo "Starting QSAR Analysis Workflow..."

# Activate virtual environment
#source venv/bin/activate

# Run preprocessing
echo "Preprocessing data..."
python3 scripts/preprocess.py

# Compute descriptors
echo "Computing molecular descriptors..."
python3 scripts/descriptors.py

# Train model
echo "Training model..."
python3 scripts/train_model.py

# Evaluate model
echo "Evaluating model..."
python3 scripts/evaluate.py

# Generate visualizations
echo "Generating visualizations..."
python3 scripts/visualize.py

echo "Workflow completed!"