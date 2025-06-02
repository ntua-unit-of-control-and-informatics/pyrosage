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
    df = pd.read_csv(f"{DATA_DIR}/{model_name}.csv")
    print(f"Dataset shape: {df.shape}")
    
    # Show target distribution
    print("Target value statistics:")
    print(df['active'].value_counts())
    
    # Plot target distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='active', data=df)
    plt.title(f'Distribution of Target Classes - {model_name}')
    plt.savefig(f"../plots/{model_name}_target_distribution.png")
    plt.close()

    # Split data with stratification to maintain class balance
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['active'])
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['active'])

    print(f"Train set: {len(train_df)}, Validation set: {len(val_df)}, Test set: {len(test_df)}")
    print(f"Class distribution in train: {train_df['active'].value_counts()}")
    print(f"Class distribution in val: {val_df['active'].value_counts()}")
    print(f"Class distribution in test: {test_df['active'].value_counts()}")

    # Check if dataset is imbalanced (use threshold of 2:1 ratio)
    class_counts = train_df['active'].value_counts()
    is_imbalanced = False
    if len(class_counts) >= 2:
        majority_count = class_counts.max()
        minority_count = class_counts.min()
        imbalance_ratio = majority_count / minority_count
        is_imbalanced = imbalance_ratio > 2.0
        print(f"Class imbalance ratio: {imbalance_ratio:.2f} - {'Imbalanced' if is_imbalanced else 'Balanced'}")

    # Apply SMOTE only if dataset is imbalanced and use_smote is enabled
    if is_imbalanced and hyperparams.get('use_smote', False):
        print("Applying SMOTE to balance training data...")
        # Extract smiles and labels
        X_smiles = train_df['smiles'].values
        y = train_df['active'].values
        
        # Create a dummy feature for SMOTE (just indices)
        X_dummy = np.arange(len(X_smiles)).reshape(-1, 1)
        
        # Apply SMOTE on the dummy feature
        smote = SMOTE(random_state=42)
        X_dummy_resampled, y_resampled = smote.fit_resample(X_dummy, y)
        
        # Get the corresponding SMILES
        smiles_resampled = X_smiles[X_dummy_resampled.ravel()]
        
        # Create new dataframe
        train_df = pd.DataFrame({
            'smiles': smiles_resampled,
            'active': y_resampled
        })
        
        print("Applied SMOTE - New training distribution:")
        print(train_df['active'].value_counts())
    else:
        print("SMOTE not applied - using original class distribution")

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
        return None, None

    # Determine number of classes from the dataset
    num_classes = df['active'].nunique()
    print(f"Number of classes: {num_classes}")

    # Create model with appropriate number of output classes
    model = AttentiveFP(
        in_channels=10,  # Enhanced atom features
        hidden_channels=hyperparams['hidden_channels'],
        out_channels=num_classes,  # Dynamic based on dataset
        edge_dim=6,  # Enhanced bond features
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
    
    # Loss function
    criterion = nn.CrossEntropyLoss()

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
    
    # Generate and save plots
    # 1. Training history
    fig = plot_training_history(history, title=f'Training History - {model_name}')
    fig.savefig(f"../plots/{model_name}_training_history.png")
    plt.close()
    
    # 2. Confusion matrix (test set)
    fig = plot_confusion_matrix(final_test_targets, final_test_preds, title=f'Confusion Matrix (Test) - {model_name}')
    fig.savefig(f"../plots/{model_name}_confusion_matrix.png")
    plt.close()
    
    # 3. ROC curve (test set) - only if binary classification
    if num_classes == 2:
        fig = plot_roc_curve(final_test_targets, final_test_preds, title=f'ROC Curve (Test) - {model_name}')
        fig.savefig(f"../plots/{model_name}_roc_curve.png")
        plt.close()
    
    # Close all plot windows
    plt.close('all')
    
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
    # Create directories for saving results
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../plots", exist_ok=True)
    
    all_datasets = [f for f in os.listdir(DATA_DIR) if isfile(join(DATA_DIR, f))]
    all_results = []
    
    for dataset_name in all_datasets:
        filename = Path(dataset_name)
        model_name = filename.name.removesuffix("".join(filename.suffixes))

        # Try different hyperparameter configurations
        dataset_results = []
        best_model = None
        best_config = None
        best_f1 = -float('inf')  # Initialize with worst possible value for F1 score

        for config in hyperparameter_configs:
            print(f"\n=== Training with configuration: {config['name']} ===")
            metrics, trained_model = train_and_evaluate(model_name, config)

            if metrics:
                # Store results for comparison
                result = {
                    "dataset": model_name,
                    "config_name": config["name"],
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "auc": metrics.get("auc", 0)  # AUC might not be present for multiclass
                }
                
                dataset_results.append(result)
                all_results.append(result)
                
                # Check if this is the best model so far (using F1 as primary metric)
                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    best_model = trained_model
                    best_config = config

        # Print comparison of results for this dataset
        if dataset_results:
            print(f"\n=== Results for {model_name} ===")
            results_df = pd.DataFrame(dataset_results)
            print(results_df)

            # Plot comparison
            plt.figure(figsize=(15, 8))

            plt.subplot(2, 2, 1)
            sns.barplot(x="config_name", y="accuracy", data=results_df)
            plt.title("Accuracy")

            plt.subplot(2, 2, 2)
            sns.barplot(x="config_name", y="precision", data=results_df)
            plt.title("Precision")

            plt.subplot(2, 2, 3)
            sns.barplot(x="config_name", y="recall", data=results_df)
            plt.title("Recall")

            plt.subplot(2, 2, 4)
            sns.barplot(x="config_name", y="f1", data=results_df)
            plt.title("F1 Score")

            plt.tight_layout()
            plt.savefig(f"../plots/{model_name}_hyperparameter_comparison.png")
            plt.close()

            # If we found a best model
            if best_model is not None and best_config is not None:
                best_config_name = best_config['name']
                best_idx = next(i for i, r in enumerate(dataset_results) if r["config_name"] == best_config_name)
                best_metrics = {
                    "accuracy": dataset_results[best_idx]["accuracy"],
                    "precision": dataset_results[best_idx]["precision"],
                    "recall": dataset_results[best_idx]["recall"],
                    "f1": dataset_results[best_idx]["f1"],
                    "auc": dataset_results[best_idx]["auc"]
                }
                
                # Add best model information
                print("\n=== Best Model ===")
                print(f"Configuration: {best_config_name}")
                print(f"Accuracy: {best_metrics['accuracy']:.4f}")
                print(f"Precision: {best_metrics['precision']:.4f}")
                print(f"Recall: {best_metrics['recall']:.4f}")
                print(f"F1 Score: {best_metrics['f1']:.4f}")
                if best_metrics['auc'] > 0:
                    print(f"AUC: {best_metrics['auc']:.4f}")
                    
                print("Hyperparameters:")
                for key, value in best_config.items():
                    if key != 'name':
                        print(f"  {key}: {value}")

                # Save only the best model with its hyperparameters
                best_model_path = f"../models/{model_name}_attentivefp_best.pt"
                torch.save({
                    'model_state_dict': best_model.state_dict(),
                    'hyperparameters': best_config
                }, best_model_path)
                print(f"Best model saved to {best_model_path}")
                
                # Save best model info to CSV
                best_model_info = {
                    'metric': ['Dataset', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'] + list(best_config.keys()),
                    'value': [model_name, best_metrics['accuracy'], best_metrics['precision'], 
                             best_metrics['recall'], best_metrics['f1'], best_metrics['auc']] + list(best_config.values())
                }
                best_model_df = pd.DataFrame(best_model_info)
                
                # Save to CSV
                with open(f"../models/{model_name}_best_model_info.csv", 'w') as f:
                    f.write("=== Best Model ===\n")
                    best_model_df.to_csv(f, index=False)

    # After processing all datasets, create an overall summary
    if all_results:
        # Create a DataFrame with all results
        all_results_df = pd.DataFrame(all_results)
        
        # Save all results to CSV
        all_results_df.to_csv("../models/classification_all_results.csv", index=False)
        
        # Create a summary of best models for each dataset
        best_models_summary = []
        for dataset in all_results_df['dataset'].unique():
            dataset_results = all_results_df[all_results_df['dataset'] == dataset]
            best_idx = dataset_results['f1'].idxmax()
            best_row = dataset_results.loc[best_idx]
            
            best_models_summary.append({
                'dataset': best_row['dataset'],
                'best_config': best_row['config_name'],
                'accuracy': best_row['accuracy'],
                'precision': best_row['precision'],
                'recall': best_row['recall'],
                'f1': best_row['f1'],
                'auc': best_row['auc']
            })
        
        best_models_df = pd.DataFrame(best_models_summary)
        best_models_df.to_csv("../models/classification_best_models_summary.csv", index=False)
        
        print("\n=== Best Models Summary ===")
        print(best_models_df)
        
        # Plot summary of best models across datasets
        plt.figure(figsize=(15, 8))
        
        metrics = ["accuracy", "precision", "recall", "f1", "auc"]
        
        for i, metric in enumerate(metrics):
            plt.subplot(1, len(metrics), i + 1)
            sns.barplot(x="dataset", y=metric, data=best_models_df)
            plt.title(f"{metric.capitalize()}")
            plt.xticks(rotation=90)
            plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig("../plots/classification_summary.png")
        plt.close()
