import os
import math
from os.path import isfile, join
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)
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
from imblearn.over_sampling import SMOTE

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

DATA_DIR = "../data/classification"

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create directories for saving results
os.makedirs("../models", exist_ok=True)
os.makedirs("../plots", exist_ok=True)

# Define evaluation metrics and visualization functions
def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    return plt

def plot_roc_curve(y_true, y_pred, title='ROC Curve'):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    return plt

def plot_pr_curve(y_true, y_pred, title='Precision-Recall Curve'):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AUC = {pr_auc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    return plt

def plot_training_history(history, title='Training History'):
    """Plot training and validation metrics over epochs"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].plot(history['test_loss'], label='Test')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Accuracy
    axes[0, 1].plot(history['train_accuracy'], label='Train')
    axes[0, 1].plot(history['val_accuracy'], label='Validation')
    axes[0, 1].plot(history['test_accuracy'], label='Test')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    
    # F1 Score
    axes[1, 0].plot(history['train_f1'], label='Train')
    axes[1, 0].plot(history['val_f1'], label='Validation')
    axes[1, 0].plot(history['test_f1'], label='Test')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    
    # AUC
    axes[1, 1].plot(history['train_auc'], label='Train')
    axes[1, 1].plot(history['val_auc'], label='Validation')
    axes[1, 1].plot(history['test_auc'], label='Test')
    axes[1, 1].set_title('ROC AUC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
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

def generate_morgan_fingerprints(smiles):
    """Generate Morgan fingerprints for SMOTE processing"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))

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
            
            # Create PyG Data object
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
            probs = torch.sigmoid(out)
            all_predictions.extend(probs.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
        
        total_loss += loss.item() * batch.num_graphs
    
    epoch_loss = total_loss / len(loader.dataset)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    binary_preds = (all_predictions > 0.5).astype(int)
    metrics = {
        'loss': epoch_loss,
        'accuracy': accuracy_score(all_targets, binary_preds),
        'precision': precision_score(all_targets, binary_preds, zero_division=0),
        'recall': recall_score(all_targets, binary_preds, zero_division=0),
        'f1': f1_score(all_targets, binary_preds, zero_division=0),
        'auc': roc_auc_score(all_targets, all_predictions)
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
            probs = torch.sigmoid(out)
            all_predictions.extend(probs.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            total_loss += loss.item() * batch.num_graphs
    
    epoch_loss = total_loss / len(loader.dataset)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    binary_preds = (all_predictions > 0.5).astype(int)
    metrics = {
        'loss': epoch_loss,
        'accuracy': accuracy_score(all_targets, binary_preds),
        'precision': precision_score(all_targets, binary_preds, zero_division=0),
        'recall': recall_score(all_targets, binary_preds, zero_division=0),
        'f1': f1_score(all_targets, binary_preds, zero_division=0),
        'auc': roc_auc_score(all_targets, all_predictions)
    }
    
    return metrics, all_predictions, all_targets

def train_and_evaluate(model_name, hyperparams):
    """Train and evaluate model with given hyperparameters"""
    print(f"\n=== Training model: {model_name} ===")
    print(f"Hyperparameters: {hyperparams}")
    
    # Load data
    df = pd.read_csv(f"../data/classification/{model_name}.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Original class distribution:")
    class_dist = df['active'].value_counts(normalize=True)
    print(class_dist)
    
    # Visualize original class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='active', data=df)
    plt.title(f'Original Class Distribution - {model_name}')
    plt.savefig(f"../plots/{model_name}_original_class_dist.png")
    plt.close()
    
    # Generate fingerprints for SMOTE
    print("\nGenerating molecular fingerprints for SMOTE...")
    fingerprints = []
    valid_indices = []
    for idx, smiles in enumerate(df['smiles']):
        fp = generate_morgan_fingerprints(smiles)
        if fp is not None:
            fingerprints.append(fp)
            valid_indices.append(idx)

    # Create feature matrix and target vector for SMOTE
    X = np.array(fingerprints)
    y = df['active'].iloc[valid_indices].values

    # Apply SMOTE
    print("Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"Original dataset shape: {X.shape}, Resampled shape: {X_resampled.shape}")

    # Create new balanced dataframe
    original_smiles = df['smiles'].iloc[valid_indices].values
    smiles_dict = dict(zip(map(tuple, X), original_smiles))

    # Get SMILES for both original and synthetic samples
    resampled_smiles = []
    for fp in X_resampled:
        fp_tuple = tuple(fp)
        if fp_tuple in smiles_dict:
            # Original molecule
            resampled_smiles.append(smiles_dict[fp_tuple])
        else:
            # Find nearest neighbor among original molecules
            distances = np.linalg.norm(X - fp, axis=1)
            nearest_idx = np.argmin(distances)
            resampled_smiles.append(original_smiles[nearest_idx])

    # Create balanced dataframe
    balanced_df = pd.DataFrame({
        'smiles': resampled_smiles,
        'active': y_resampled
    })

    # Print and visualize balanced class distribution
    print("\nBalanced class distribution:")
    balanced_dist = balanced_df['active'].value_counts(normalize=True)
    print(balanced_dist)
    
    plt.figure(figsize=(8, 6))
    sns.countplot(x='active', data=balanced_df)
    plt.title(f'Balanced Class Distribution After SMOTE - {model_name}')
    plt.savefig(f"../plots/{model_name}_balanced_class_dist.png")
    plt.close()

    # Split balanced data
    train_df, test_df = train_test_split(balanced_df, test_size=0.2, random_state=42, stratify=balanced_df['active'])
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['active'])

    print(f"\nTrain set: {len(train_df)}, Validation set: {len(val_df)}, Test set: {len(test_df)}")

    # Create datasets and dataloaders
    train_dataset = MoleculeDataset(train_df)
    val_dataset = MoleculeDataset(val_df)
    test_dataset = MoleculeDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'])

    # Check if any dataloaders are empty
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        print("Error: One or more datasets are empty!")
        return None

    # Create model
    model = AttentiveFP(
        in_channels=10,  # Updated for enhanced atom features
        hidden_channels=hyperparams['hidden_channels'],
        out_channels=1,  # 1 output for binary classification
        edge_dim=6,  # Updated for enhanced bond features
        num_layers=hyperparams['num_layers'],
        num_timesteps=hyperparams['num_timesteps'],
        dropout=hyperparams['dropout']
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
        lr=hyperparams['learning_rate'],
        weight_decay=hyperparams['weight_decay']
    )

    # Loss function for binary classification
    criterion = nn.BCEWithLogitsLoss()

    # Learning rate scheduler
    num_training_steps = hyperparams['epochs'] * len(train_loader)
    num_warmup_steps = num_training_steps // 10
    
    def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # Early stopping
    early_stopping = EarlyStopping(patience=hyperparams['patience'])
    
    # Training history
    history = defaultdict(list)

    # Training loop
    for epoch in range(hyperparams['epochs']):
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
            history[f'train_{k}'].append(v)
        for k, v in val_metrics.items():
            history[f'val_{k}'].append(v)
        for k, v in test_metrics.items():
            history[f'test_{k}'].append(v)
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{hyperparams['epochs']}")
        print("Train -", ' '.join([f"{k}: {v:.4f}" for k, v in train_metrics.items()]))
        print("Val   -", ' '.join([f"{k}: {v:.4f}" for k, v in val_metrics.items()]))
        print("Test  -", ' '.join([f"{k}: {v:.4f}" for k, v in test_metrics.items()]))
        
        # Early stopping check
        early_stopping(val_metrics['loss'], model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            # Load best model
            model.load_state_dict(early_stopping.best_state_dict)
            break
    
    # Load best model for final evaluation
    if early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)
    
    # Final evaluation
    final_train_metrics, final_train_preds, final_train_targets = evaluate(model, train_loader, criterion, device)
    final_val_metrics, final_val_preds, final_val_targets = evaluate(model, val_loader, criterion, device)
    final_test_metrics, final_test_preds, final_test_targets = evaluate(model, test_loader, criterion, device)
    
    print("\n=== Final Evaluation ===")
    print("Train -", ' '.join([f"{k}: {v:.4f}" for k, v in final_train_metrics.items()]))
    print("Val   -", ' '.join([f"{k}: {v:.4f}" for k, v in final_val_metrics.items()]))
    print("Test  -", ' '.join([f"{k}: {v:.4f}" for k, v in final_test_metrics.items()]))
    
    # Classification report
    print("\nClassification Report (Test Set):")
    binary_preds = (final_test_preds > 0.5).astype(int)
    print(classification_report(final_test_targets, binary_preds))
    
    # Generate and save plots
    # 1. Training history
    fig = plot_training_history(history, title=f'Training History - {model_name}')
    fig.savefig(f"../plots/{model_name}_training_history.png")
    
    # 2. ROC curve
    fig = plot_roc_curve(final_test_targets, final_test_preds, title=f'ROC Curve (Test) - {model_name}')
    fig.savefig(f"../plots/{model_name}_roc_curve.png")
    
    # 3. Precision-Recall curve
    fig = plot_pr_curve(final_test_targets, final_test_preds, title=f'PR Curve (Test) - {model_name}')
    fig.savefig(f"../plots/{model_name}_pr_curve.png")
    
    # 4. Confusion Matrix
    fig = plot_confusion_matrix(final_test_targets, binary_preds, title=f'Confusion Matrix (Test) - {model_name}')
    fig.savefig(f"../plots/{model_name}_confusion_matrix.png")
    
    # Save the model and results
    save_path = f"../models/{model_name}_attentivefp.pt"
    results = {
        'model_state_dict': model.state_dict(),
        'hyperparams': hyperparams,
        'train_metrics': final_train_metrics,
        'val_metrics': final_val_metrics,
        'test_metrics': final_test_metrics,
        'history': dict(history)
    }
    torch.save(results, save_path)
    print(f"Model and results saved to {save_path}")
    
    # Close all plot windows
    plt.close('all')
    
    return final_test_metrics

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
    }
]

if __name__ == "__main__":
    all_datasets = [f for f in os.listdir(DATA_DIR) if isfile(join(DATA_DIR, f))]
    all_results = []
    for dataset_name in all_datasets:
        filename = Path(dataset_name)
        model_name = filename.name.removesuffix("".join(filename.suffixes))

        # Run training and evaluation for each dataset
        print(f"\n========= Processing Dataset: {model_name} =========")

        # Use the baseline configuration for all datasets
        config = hyperparameter_configs[0]
        metrics = train_and_evaluate(model_name, config)

        if metrics:
            all_results.append(
                {
                    "dataset": model_name,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "auc": metrics["auc"],
                }
            )

    # Create and save summary table
    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\n=== Summary of Results ===")
        print(results_df)

        # Save summary to CSV
        results_df.to_csv("../models/classification_results_summary.csv", index=False)

        # Plot summary of metrics across datasets
        plt.figure(figsize=(15, 8))

        metrics = ["accuracy", "precision", "recall", "f1", "auc"]

        for i, metric in enumerate(metrics):
            plt.subplot(1, len(metrics), i + 1)
            sns.barplot(x="dataset", y=metric, data=results_df)
            plt.title(f"{metric.capitalize()}")
            plt.xticks(rotation=90)
            plt.ylim(0, 1)

        plt.tight_layout()
        plt.savefig("../plots/classification_summary.png")
        plt.close()
