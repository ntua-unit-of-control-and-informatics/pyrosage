import logging
import os
import math
from os.path import isfile, join
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentiveFP

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

DATA_DIR = "../data/regression"

# Create directories for saving results
os.makedirs("../models", exist_ok=True)
os.makedirs("../plots", exist_ok=True)

# Define evaluation metrics and visualization functions
def plot_scatter(y_true, y_pred, title='Predicted vs Actual'):
    """Plot scatter plot of predicted vs actual values"""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)

    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    return plt

def plot_residuals(y_true, y_pred, title='Residuals Plot'):
    """Plot residuals"""
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(title)
    return plt

def plot_training_history(history, title='Training History'):
    """Plot training and validation metrics over epochs"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].plot(history['test_loss'], label='Test')
    axes[0, 0].set_title('Loss (MSE)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()

    # MAE
    axes[0, 1].plot(history['train_mae'], label='Train')
    axes[0, 1].plot(history['val_mae'], label='Validation')
    axes[0, 1].plot(history['test_mae'], label='Test')
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()

    # RMSE
    axes[1, 0].plot(history['train_rmse'], label='Train')
    axes[1, 0].plot(history['val_rmse'], label='Validation')
    axes[1, 0].plot(history['test_rmse'], label='Test')
    axes[1, 0].set_title('Root Mean Squared Error')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].legend()

    # R²
    axes[1, 1].plot(history['train_r2'], label='Train')
    axes[1, 1].plot(history['val_r2'], label='Validation')
    axes[1, 1].plot(history['test_r2'], label='Test')
    axes[1, 1].set_title('R² Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].legend()

    plt.suptitle(title)
    plt.tight_layout()
    return plt

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_state_dict = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state_dict = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class MoleculeDataset(Dataset):
    """Dataset for molecular graphs"""
    def __init__(self, df, smiles_col='smiles', target_col='active'):
        super().__init__()
        self.df = df
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.data_list = self._prepare_data()

    def _prepare_data(self):
        data_list = []
        for idx, row in self.df.iterrows():
            mol = Chem.MolFromSmiles(row[self.smiles_col])
            if mol is None:
                continue

            # Enhanced atom features
            atom_features = []
            for atom in mol.GetAtoms():
                features = [
                    atom.GetAtomicNum(),
                    atom.GetTotalDegree(),
                    atom.GetFormalCharge(),
                    atom.GetTotalNumHs(),
                    atom.GetNumRadicalElectrons(),
                    int(atom.GetIsAromatic()),
                    int(atom.IsInRing()),
                    # Hybridization as one-hot
                    int(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP),
                    int(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2),
                    int(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3)
                ]
                atom_features.append(features)

            x = torch.tensor(atom_features, dtype=torch.float)

            # Enhanced bond features
            edges_list = []
            edge_features = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edges_list.extend([[i, j], [j, i]])

                features = [
                    # Bond type as one-hot
                    int(bond.GetBondType() == Chem.rdchem.BondType.SINGLE),
                    int(bond.GetBondType() == Chem.rdchem.BondType.DOUBLE),
                    int(bond.GetBondType() == Chem.rdchem.BondType.TRIPLE),
                    int(bond.GetBondType() == Chem.rdchem.BondType.AROMATIC),
                    # Additional features
                    int(bond.GetIsConjugated()),
                    int(bond.IsInRing())
                ]
                edge_features.extend([features, features])

            if not edges_list:  # Skip molecules with no bonds
                continue

            edge_index = torch.tensor(edges_list, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)

            # Create PyG Data object - note target is now a float for regression
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([float(row[self.target_col])], dtype=torch.float),
            )
            data_list.append(data)
        return data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

def train_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.edge_attr, batch=batch.batch)
        out = out.squeeze()
        target = batch.y.squeeze()

        loss = criterion(out, target)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Store predictions and targets
        with torch.no_grad():
            all_predictions.extend(out.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

        total_loss += loss.item() * batch.num_graphs

    epoch_loss = total_loss / len(loader.dataset)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Calculate regression metrics
    metrics = {
        'loss': epoch_loss,
        'mae': mean_absolute_error(all_targets, all_predictions),
        'rmse': np.sqrt(mean_squared_error(all_targets, all_predictions)),
        'r2': r2_score(all_targets, all_predictions)
    }

    return metrics, all_predictions, all_targets

def evaluate(model, loader, criterion, device):
    """Evaluate model on a data loader"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(
                batch.x, batch.edge_index, batch.edge_attr,
                batch=batch.batch
            )
            out = out.squeeze()
            target = batch.y.squeeze()
            loss = criterion(out, target)

            # Store predictions and targets
            all_predictions.extend(out.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            total_loss += loss.item() * batch.num_graphs

    epoch_loss = total_loss / len(loader.dataset)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Calculate regression metrics
    metrics = {
        'loss': epoch_loss,
        'mae': mean_absolute_error(all_targets, all_predictions),
        'rmse': np.sqrt(mean_squared_error(all_targets, all_predictions)),
        'r2': r2_score(all_targets, all_predictions)
    }

    return metrics, all_predictions, all_targets

def train_and_evaluate(model_name, hyperparams):
    """Train and evaluate model with given hyperparameters"""
    print(f"\n=== Training model: {model_name} ===")
    print(f"Hyperparameters: {hyperparams}")

    # Load data
    df = pd.read_csv(f"{DATA_DIR}/{model_name}.csv")
    print(f"Dataset shape: {df.shape}")

    # Show target distribution
    print("Target value statistics:")
    print(df["active"].describe())

    # Plot target distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df["active"], kde=True)
    plt.title(f"Distribution of Target Values - {model_name}")
    plt.savefig(f"../plots/{model_name}_target_distribution.png")
    plt.close()

    # Split data - no need for stratification in regression
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    print(
        f"Train set: {len(train_df)}, Validation set: {len(val_df)}, Test set: {len(test_df)}"
    )

    # Create datasets and dataloaders
    train_dataset = MoleculeDataset(train_df)
    val_dataset = MoleculeDataset(val_df)
    test_dataset = MoleculeDataset(test_df)

    train_loader = DataLoader(
        train_dataset, batch_size=hyperparams["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=hyperparams["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=hyperparams["batch_size"])

    # Check if any dataloaders are empty
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        print("Error: One or more datasets are empty!")
        return None, None

    # Create model
    model = AttentiveFP(
        in_channels=10,  # Updated for enhanced atom features
        hidden_channels=hyperparams["hidden_channels"],
        out_channels=1,  # 1 output for regression
        edge_dim=6,  # Updated for enhanced bond features
        num_layers=hyperparams["num_layers"],
        num_timesteps=hyperparams["num_timesteps"],
        dropout=hyperparams["dropout"],
    ).to(device)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    model.apply(init_weights)

    # Set up optimizer and criterion
    optimizer = optim.AdamW(
        model.parameters(),
        lr=hyperparams["learning_rate"],
        weight_decay=hyperparams["weight_decay"],
    )

    # Loss function - use MSE for regression
    criterion = nn.MSELoss()

    # Learning rate scheduler
    num_training_steps = hyperparams["epochs"] * len(train_loader)
    num_warmup_steps = num_training_steps // 10

    def get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    ):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=hyperparams["patience"])

    # Training history
    history = defaultdict(list)

    # Training loop
    for epoch in range(hyperparams["epochs"]):
        # Training
        train_metrics, train_preds, train_targets = train_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )

        # Validation
        val_metrics, val_preds, val_targets = evaluate(
            model, val_loader, criterion, device
        )

        # Test
        test_metrics, test_preds, test_targets = evaluate(
            model, test_loader, criterion, device
        )

        # Update history
        for k, v in train_metrics.items():
            history[f"train_{k}"].append(v)
        for k, v in val_metrics.items():
            history[f"val_{k}"].append(v)
        for k, v in test_metrics.items():
            history[f"test_{k}"].append(v)

        # Print metrics
        print(f"\nEpoch {epoch+1}/{hyperparams['epochs']}")
        print("Train -", " ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()]))
        print("Val   -", " ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()]))
        print("Test  -", " ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items()]))

        # Early stopping check
        early_stopping(val_metrics["loss"], model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            # Load best model
            model.load_state_dict(early_stopping.best_state_dict)
            break

    # Load best model for final evaluation
    if early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)

    # Final evaluation
    final_train_metrics, final_train_preds, final_train_targets = evaluate(
        model, train_loader, criterion, device
    )
    final_val_metrics, final_val_preds, final_val_targets = evaluate(
        model, val_loader, criterion, device
    )
    final_test_metrics, final_test_preds, final_test_targets = evaluate(
        model, test_loader, criterion, device
    )

    print("\n=== Final Evaluation ===")
    print(
        "Train -", " ".join([f"{k}: {v:.4f}" for k, v in final_train_metrics.items()])
    )
    print("Val   -", " ".join([f"{k}: {v:.4f}" for k, v in final_val_metrics.items()]))
    print("Test  -", " ".join([f"{k}: {v:.4f}" for k, v in final_test_metrics.items()]))

    # Generate and save plots
    # 1. Training history
    fig = plot_training_history(history, title=f"Training History - {model_name}")
    fig.savefig(f"../plots/{model_name}_training_history.png")
    plt.close()

    # 2. Scatter plot of predicted vs actual (test set)
    fig = plot_scatter(
        final_test_targets,
        final_test_preds,
        title=f"Predicted vs Actual (Test) - {model_name}",
    )
    fig.savefig(f"../plots/{model_name}_scatter_plot.png")
    plt.close()

    # 3. Residuals plot
    fig = plot_residuals(
        final_test_targets,
        final_test_preds,
        title=f"Residuals Plot (Test) - {model_name}",
    )
    fig.savefig(f"../plots/{model_name}_residuals.png")
    plt.close()

    # Return metrics and the trained model (not saving it yet)
    return final_test_metrics, model


# Define hyperparameter configurations to try
hyperparameter_configs = [
    {
        'name': 'baseline',
        'hidden_channels': 64,
        'num_layers': 2,
        'num_timesteps': 2,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'batch_size': 32,
        'epochs': 50,
        'patience': 10
    },
    {
        'name': 'larger_model',
        'hidden_channels': 128,
        'num_layers': 3,
        'num_timesteps': 3,
        'dropout': 0.1,
        'learning_rate': 0.0005,
        'weight_decay': 1e-4,
        'batch_size': 32,
        'epochs': 50,
        'patience': 10
    },
    {
        'name': 'deeper_model',
        'hidden_channels': 64,
        'num_layers': 4,
        'num_timesteps': 4,
        'dropout': 0.3,
        'learning_rate': 0.0005,
        'weight_decay': 1e-4,
        'batch_size': 32,
        'epochs': 50,
        'patience': 10
    }
]


if __name__ == "__main__":
    all_datasets = [f for f in os.listdir(DATA_DIR) if isfile(join(DATA_DIR, f))]
    for dataset_name in all_datasets:
        filename = Path(dataset_name)
        model_name = filename.name.removesuffix("".join(filename.suffixes))

        # Try different hyperparameter configurations
        results = []
        best_model = None
        best_config = None
        best_r2 = -float("inf")  # Initialize with worst possible value

        for config in hyperparameter_configs:
            print(f"\n=== Training with configuration: {config['name']} ===")
            metrics, trained_model = train_and_evaluate(model_name, config)

            if metrics:
                # Store results for comparison
                results.append(
                    {
                        "config_name": config["name"],
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "r2": metrics["r2"],
                    }
                )

                # Check if this is the best model so far
                if metrics["r2"] > best_r2:
                    best_r2 = metrics["r2"]
                    best_model = trained_model
                    best_config = config

        # Print comparison of results
        if results:
            print("\n=== Hyperparameter Comparison ===")
            results_df = pd.DataFrame(results)
            print(results_df)

            # Plot comparison
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            sns.barplot(x="config_name", y="rmse", data=results_df)
            plt.title("RMSE (lower is better)")

            plt.subplot(1, 3, 2)
            sns.barplot(x="config_name", y="mae", data=results_df)
            plt.title("MAE (lower is better)")

            plt.subplot(1, 3, 3)
            sns.barplot(x="config_name", y="r2", data=results_df)
            plt.title("R² (higher is better)")

            plt.tight_layout()
            plt.savefig(f"../plots/{model_name}_hyperparameter_comparison.png")
            plt.close()

            # If we found a best model
            if best_model is not None and best_config is not None:
                best_config_name = best_config["name"]
                best_idx = next(
                    i
                    for i, r in enumerate(results)
                    if r["config_name"] == best_config_name
                )
                best_rmse = results[best_idx]["rmse"]
                best_mae = results[best_idx]["mae"]

                # Add best model information
                print("\n=== Best Model ===")
                print(f"Configuration: {best_config_name}")
                print(f"RMSE: {best_rmse:.4f}, MAE: {best_mae:.4f}, R²: {best_r2:.4f}")
                print("Hyperparameters:")
                for key, value in best_config.items():
                    if key != "name":
                        print(f"  {key}: {value}")

                # Save only the best model with its hyperparameters
                best_model_path = f"../models/{model_name}_attentivefp_best.pt"
                torch.save(
                    {
                        "model_state_dict": best_model.state_dict(),
                        "hyperparameters": best_config,
                    },
                    best_model_path,
                )
                print(f"Best model saved to {best_model_path}")
