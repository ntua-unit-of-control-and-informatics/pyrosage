# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Pyrosage** project - a machine learning codebase for predicting environmental and toxicity properties of chemical compounds using Graph Neural Networks (specifically AttentiveFP). The project focuses on developing predictive models for virtual screening and safer chemical design.

## Key Technologies

- **PyTorch Geometric**: Graph neural network library for molecular graphs
- **AttentiveFP**: Attention-based fingerprint GNN model for molecular property prediction
- **RDKit**: Chemical informatics toolkit for molecular processing
- **SMILES**: Molecular representation format used throughout

## Architecture Overview

### Core Components

1. **code/predict.py**: Main prediction interface that loads trained models and makes predictions on SMILES strings
2. **code/train/molecule_dataset.py**: Custom PyTorch Dataset class that converts SMILES to molecular graphs with enhanced atom/bond features
3. **code/train/train_all_models_attentivefp_classification.py**: Training pipeline for binary classification tasks (AMES, Endocrine Disruption, Eye Irritation)
4. **code/train/train_all_models_attentivefp_regression.py**: Training pipeline for regression tasks (LC50, LD50)

### Data Processing Pipeline

- SMILES strings → RDKit molecular objects → PyTorch Geometric Data objects
- Enhanced atom features (10 dimensions): atomic number, degree, formal charge, hydrogens, radical electrons, aromaticity, ring membership, hybridization (one-hot)
- Enhanced bond features (6 dimensions): bond type (one-hot), conjugation, ring membership
- Automatic train/validation/test splits with SMOTE for classification imbalance handling

### Model Architecture

- AttentiveFP models with configurable hyperparameters
- Multiple configurations tested: baseline, larger_model, deeper_model, class_imbalance_focused
- Early stopping with F1 monitoring for imbalanced datasets
- Class weighting for highly imbalanced classification tasks

## Dataset Structure

The project includes several toxicity/environmental datasets in `code/` subfolders:
- **AMES/**: Mutagenicity prediction
- **EndocrineDisruption/**: Endocrine disruption endpoints (NR-AR, NR-AhR, NR-ER, NR-aromatase)
- **EyeIrritation/**: Eye corrosion and irritation
- **LC50/**: Aquatic toxicity (regression)
- **LD50/**: Acute oral toxicity (regression)

Each dataset folder contains CSV files and Jupyter notebooks for data transformation.

## Common Commands

### Dependencies Setup
```bash
# Install required dependencies
pip install -r requirements.txt
```

### Training Models
```bash
# Train all classification models with hyperparameter search
cd code/train
python train_all_models_attentivefp_classification.py

# Train all regression models with hyperparameter search  
python train_all_models_attentivefp_regression.py
```

### Making Predictions
```bash
# Run prediction script (loads all trained models)
cd code
python predict.py
```

### Data Processing
```bash
# Individual dataset transformation notebooks are in each dataset folder
cd code
jupyter notebook AMES/retrieve_dataset.ipynb
jupyter notebook EndocrineDisruption/transform_datasets.ipynb
jupyter notebook LC50/transform_dataset.ipynb
jupyter notebook EyeIrritation/transform_datasets.ipynb
jupyter notebook LD50/retrieve_dataset.ipynb
```

## Model Storage

- Classification models saved to: `models/classification/`
- Regression models saved to: `models/regression/`
- Best models saved with suffix `_best.pt` containing both state_dict and hyperparameters
- Model info saved as CSV files with suffix `_best_model_info.csv`
- Summary files: `classification_all_results.csv` and `classification_best_models_summary.csv`

## File Paths and Working Directory

When working with this codebase, be aware of relative path dependencies:
- Training scripts (in `code/train/`) expect to save models to `../../models/`
- Prediction script (in `code/`) expects models in `../models/`
- Data loading uses `DATA_DIR = "../../data/classification"` from training scripts
- Always run scripts from their intended directories to maintain correct paths

## Development Notes

- Models expect specific input dimensions: in_channels=10 (atom features), edge_dim=6 (bond features)
- MoleculeDataset class handles SMILES → PyG Data conversion with feature engineering
- CUDA support auto-detected and used when available
- Extensive plotting and evaluation metrics generated during training (saved to `plots/`)
- Class imbalance handling with SMOTE oversampling and weighted loss functions
- Early stopping prevents overfitting with configurable patience and monitoring metrics
- Hyperparameter search includes: baseline, class_imbalance_focused, larger_model, deeper_model