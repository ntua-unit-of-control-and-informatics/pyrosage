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

from molecule_dataset import MoleculeDataset

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

DATA_DIR = "../../data/classification"

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create directories for saving results
os.makedirs("../../models/classification", exist_ok=True)
os.makedirs("../../plots", exist_ok=True)

# Define evaluation metrics and visualization functions

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    return fig

def plot_roc_curve(y_true, y_pred, title='ROC Curve'):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    return fig

def plot_pr_curve(y_true, y_pred, title='Precision-Recall Curve'):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AUC = {pr_auc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    return fig

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
    return fig  # Return the figure object, not plt

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0, monitor='loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = float('inf') if monitor == 'loss' else float('-inf')
        self.early_stop = False
        self.best_state_dict = None
        self.monitor = monitor
        self.is_loss = monitor == 'loss'
        
    def __call__(self, metrics, model):
        current_score = metrics[self.monitor]
        
        if self.is_loss:
            score = -current_score
            delta = -self.min_delta
        else:
            score = current_score
            delta = self.min_delta
        
        if score > self.best_score + delta:
            self.best_score = score
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

def train_epoch(
    model, train_loader, val_loader, optimizer, criterion, device, scheduler=None
):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.edge_attr, batch=batch.batch)

        # Make sure both have the right shape
        out_squeezed = out.squeeze(-1)  # Remove last dimension if it's 1
        target = batch.y.squeeze(-1)  # Remove last dimension if it's 1

        # Handle case where batch size is 1
        if out_squeezed.dim() == 0:
            out_squeezed = out_squeezed.unsqueeze(0)
        if target.dim() == 0:
            target = target.unsqueeze(0)

        loss = criterion(out_squeezed, target)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Store predictions and targets
        with torch.no_grad():
            probs = torch.sigmoid(out)
            all_predictions.extend(probs.cpu().numpy())
            all_targets.extend(batch.y.cpu().numpy())

        total_loss += loss.item() * batch.num_graphs

    # Validation for scheduler - make sure it gets a dictionary
    val_metrics, _, _ = evaluate(model, val_loader, criterion, device)

    if scheduler is not None:
        scheduler.step(val_metrics["loss"])

    epoch_loss = total_loss / len(train_loader.dataset)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Calculate metrics
    binary_preds = (all_predictions > 0.5).astype(int)
    metrics = {
        "loss": epoch_loss,
        "accuracy": accuracy_score(all_targets, binary_preds),
        "precision": precision_score(all_targets, binary_preds, zero_division=0),
        "recall": recall_score(all_targets, binary_preds, zero_division=0),
        "f1": f1_score(all_targets, binary_preds, zero_division=0),
        "auc": roc_auc_score(all_targets, all_predictions),
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
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch=batch.batch)

            # Make sure both have the right shape before computing loss
            out_squeezed = out.squeeze(-1)  # Remove last dimension if it's 1
            target = batch.y.squeeze(-1)  # Remove last dimension if it's 1

            # Handle case where batch size is 1
            if out_squeezed.dim() == 0:
                out_squeezed = out_squeezed.unsqueeze(0)
            if target.dim() == 0:
                target = target.unsqueeze(0)

            loss = criterion(out_squeezed, target)

            # Store predictions and targets
            probs = torch.sigmoid(out)
            all_predictions.extend(probs.cpu().numpy())
            all_targets.extend(batch.y.cpu().numpy())

            total_loss += loss.item() * batch.num_graphs

    epoch_loss = total_loss / len(loader.dataset)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Calculate metrics
    binary_preds = (all_predictions > 0.5).astype(int)
    metrics = {
        "loss": epoch_loss,
        "accuracy": accuracy_score(all_targets, binary_preds),
        "precision": precision_score(all_targets, binary_preds, zero_division=0),
        "recall": recall_score(all_targets, binary_preds, zero_division=0),
        "f1": f1_score(all_targets, binary_preds, zero_division=0),
        "auc": roc_auc_score(all_targets, all_predictions),
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

    # Split data with stratification to maintain class balance before SMOTE
    # SMOTE should only be applied to the training set
    original_train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['active'])
    original_train_df, val_df = train_test_split(original_train_df, test_size=0.2, random_state=42, stratify=original_train_df['active'])

    print(f"Original Train set: {len(original_train_df)}, Validation set: {len(val_df)}, Test set: {len(test_df)}")
    print(f"Class distribution in original train: {original_train_df['active'].value_counts()}")
    print(f"Class distribution in val: {val_df['active'].value_counts()}")
    print(f"Class distribution in test: {test_df['active'].value_counts()}")

    # Apply SMOTE to the training data, similar to the notebook
    print("\nApplying SMOTE to training data...")
    train_fingerprints = []
    train_valid_indices = []
    train_original_smiles_list = []

    for idx, row in original_train_df.iterrows():
        fp = generate_morgan_fingerprints(row['smiles']) # Assuming nBits=1024, radius=2 from notebook context
        if fp is not None:
            train_fingerprints.append(fp)
            train_valid_indices.append(idx) # Store original dataframe index
            train_original_smiles_list.append(row['smiles'])

    if not train_fingerprints:
        print("Warning: No valid fingerprints generated for SMOTE. Using original training data.")
        train_df = original_train_df.copy()
    else:
        X_train_fp = np.array(train_fingerprints)
        # Get labels corresponding to the valid fingerprints
        y_train_original = original_train_df.loc[train_valid_indices, 'active'].values 
        
        print(f"Class distribution before SMOTE on {len(y_train_original)} valid training samples:")
        print(pd.Series(y_train_original).value_counts(normalize=True))

        smote = SMOTE(random_state=42)
        try:
            X_train_resampled_fp, y_train_resampled = smote.fit_resample(X_train_fp, y_train_original)
            print(f"Class distribution after SMOTE (Fingerprints: {X_train_resampled_fp.shape[0]}):")
            print(pd.Series(y_train_resampled).value_counts(normalize=True))

            # Reconstruct balanced dataframe for training
            # Map original fingerprints back to original SMILES
            fp_to_smiles_map = {tuple(fp): smi for fp, smi in zip(X_train_fp, train_original_smiles_list)}
            
            resampled_smiles_for_train = []
            for fp_resampled in X_train_resampled_fp:
                fp_tuple = tuple(fp_resampled)
                if fp_tuple in fp_to_smiles_map:
                    resampled_smiles_for_train.append(fp_to_smiles_map[fp_tuple])
                else:
                    # For synthetic samples, find nearest original fingerprint and use its SMILES
                    distances = np.linalg.norm(X_train_fp - fp_resampled, axis=1)
                    nearest_idx = np.argmin(distances)
                    resampled_smiles_for_train.append(train_original_smiles_list[nearest_idx])
            
            train_df = pd.DataFrame({
                'smiles': resampled_smiles_for_train,
                'active': y_train_resampled
            })
            print(f"SMOTE applied. New training set size: {len(train_df)}")

        except Exception as e:
            print(f"Error applying SMOTE: {e}. Using original training data.")
            train_df = original_train_df.copy() # Fallback to original if SMOTE fails

    print(f"Final Train set size: {len(train_df)}")

    # Create datasets and dataloaders
    train_dataset = MoleculeDataset(train_df, smiles_col='smiles', target_col='active') # Use the SMOTE-balanced train_df
    val_dataset = MoleculeDataset(val_df, smiles_col='smiles', target_col='active')
    test_dataset = MoleculeDataset(test_df, smiles_col='smiles', target_col='active')

    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'])

    # Check if any dataloaders are empty
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        print("Error: One or more datasets are empty after SMOTE/processing!")
        # Log dataset sizes for debugging
        print(f"Train dataset length: {len(train_dataset)}")
        print(f"Validation dataset length: {len(val_dataset)}")
        print(f"Test dataset length: {len(test_dataset)}")
        return None, None

    # Inside train_and_evaluate, before model instantiation
    print(f"DEBUG: Initializing AttentiveFP with in_channels=7, hidden_channels={hyperparams['hidden_channels']}, edge_dim=5")
    
    # Use F1 score for early stopping in imbalanced datasets
    if df['active'].value_counts(normalize=True).min() < 0.25:  # If minority class < 25%
        early_stopping = EarlyStopping(patience=hyperparams['patience'], monitor='f1')
        print("Using F1 score for early stopping due to class imbalance")
    else:
        early_stopping = EarlyStopping(patience=hyperparams['patience'], monitor='loss')
        print("Using loss for early stopping")
    
    model = AttentiveFP(
        in_channels=10,
        hidden_channels=hyperparams['hidden_channels'],
        out_channels=1,
        edge_dim=6,
        num_layers=hyperparams['num_layers'],
        num_timesteps=hyperparams['num_timesteps'],
        dropout=hyperparams['dropout']
    ).to(device)

    # Optionally, print model's perception of its parameters if accessible
    # print(f"DEBUG: Model initialized with model.hidden_channels={model.hidden_channels}, model.in_channels={model.original_in_channels_attr_name_if_exists}") # Adjust attribute names
    # print(f"DEBUG: model.gate_conv.lin1.weight.shape = {model.gate_conv.lin1.weight.shape}")
    # Expected shape for model.gate_conv.lin1.weight should be (hyperparams['hidden_channels'], hyperparams['hidden_channels'] + 5)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    model.apply(init_weights)

    # Set up optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=hyperparams['learning_rate'],
        weight_decay=hyperparams['weight_decay']
    )
    # After creating datasets and before model initialization

    # Calculate class weights for imbalanced datasets (even with SMOTE applied)
    # This helps the model focus on minority class during training
    if df['active'].nunique() == 2:  # Binary classification
        class_counts = df['active'].value_counts()
        total_samples = len(df)
        weight_for_0 = 1.0 / (class_counts[0] / total_samples)
        weight_for_1 = 1.0 / (class_counts[1] / total_samples)

        # Normalize weights
        weight_sum = weight_for_0 + weight_for_1
        weight_for_0 = weight_for_0 / weight_sum
        weight_for_1 = weight_for_1 / weight_sum

        # Check if highly imbalanced (one class < 20%)
        if min(class_counts) / total_samples < 0.2:
            # Apply class weighting for highly imbalanced datasets
            pos_weight = torch.tensor([weight_for_1 / weight_for_0]).to(device)
            print(f"Using positive class weight: {pos_weight.item():.4f} for imbalanced dataset")
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()  # No weighting for balanced datasets
    else:
        criterion = nn.BCEWithLogitsLoss()  # Default for non-binary classification

    # Learning rate scheduler
    num_training_steps = hyperparams['epochs'] * len(train_loader) if len(train_loader) > 0 else hyperparams['epochs']
    num_warmup_steps = num_training_steps // 10 if num_training_steps > 0 else 0
    

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training history
    history = defaultdict(list)

    # Training loop
    for epoch in range(hyperparams['epochs']):
        if len(train_loader) == 0:
            print("Skipping training epoch as train_loader is empty.")
            # Populate history with dummy values if needed or handle appropriately
            for k_metric in ['loss', 'accuracy', 'precision', 'recall', 'f1', 'auc']:
                history[f'train_{k_metric}'].append(float('nan'))
            # Still run validation and test
        else:
            # Training
            train_metrics, train_preds, train_targets = train_epoch(
                model, train_loader, val_loader, optimizer, criterion, device
            )

            for k, v in train_metrics.items():
                history[f'train_{k}'].append(v)
        
        # Validation
        if len(val_loader) > 0:
            val_metrics, val_preds, val_targets = evaluate(
                model, val_loader, criterion, device
            )
            for k, v in val_metrics.items():
                history[f'val_{k}'].append(v)
        else: # Handle empty val_loader
            val_metrics = {'loss': float('nan'), 'auc': float('nan')} # for early stopping
            for k_metric in ['loss', 'accuracy', 'precision', 'recall', 'f1', 'auc']:
                 history[f'val_{k_metric}'].append(float('nan'))


        # Test
        if len(test_loader) > 0:
            test_metrics, test_preds, test_targets = evaluate(
                model, test_loader, criterion, device
            )
            for k, v in test_metrics.items():
                history[f'test_{k}'].append(v)
        else: # Handle empty test_loader
            for k_metric in ['loss', 'accuracy', 'precision', 'recall', 'f1', 'auc']:
                 history[f'test_{k_metric}'].append(float('nan'))

        # Print metrics
        print(f"\nEpoch {epoch+1}/{hyperparams['epochs']}")
        if len(train_loader)>0: print("Train -", ' '.join([f"{k}: {v:.4f}" for k, v in train_metrics.items()]))
        if len(val_loader)>0: print("Val   -", ' '.join([f"{k}: {v:.4f}" for k, v in val_metrics.items()]))
        if len(test_loader)>0: print("Test  -", ' '.join([f"{k}: {v:.4f}" for k, v in test_metrics.items()]))
        
        # Early stopping check 
        if len(val_loader) > 0:
            early_stopping(val_metrics, model)  # Pass the whole metrics dict
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                if early_stopping.best_state_dict:
                    model.load_state_dict(early_stopping.best_state_dict)
                break
    
    # Load best model for final evaluation
    if early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)
    
    # Final evaluation
    final_train_metrics, final_val_metrics, final_test_metrics = {}, {}, {}
    final_train_preds, final_val_preds, final_test_preds = np.array([]), np.array([]), np.array([])
    final_train_targets, final_val_targets, final_test_targets = np.array([]), np.array([]), np.array([])

    if len(train_loader) > 0:
        final_train_metrics, final_train_preds, final_train_targets = evaluate(model, train_loader, criterion, device)
        print("\n=== Final Evaluation (Train) ===")
        print("Train -", ' '.join([f"{k}: {v:.4f}" for k, v in final_train_metrics.items()]))
    if len(val_loader) > 0:
        final_val_metrics, final_val_preds, final_val_targets = evaluate(model, val_loader, criterion, device)
        print("\n=== Final Evaluation (Validation) ===")
        print("Val   -", ' '.join([f"{k}: {v:.4f}" for k, v in final_val_metrics.items()]))
    if len(test_loader) > 0:
        final_test_metrics, final_test_preds, final_test_targets = evaluate(model, test_loader, criterion, device)
        print("\n=== Final Evaluation (Test) ===")
        print("Test  -", ' '.join([f"{k}: {v:.4f}" for k, v in final_test_metrics.items()]))
    
    # Generate and save plots only if data is available
    if len(history['train_loss']) > 0 : # Check if training actually ran
        fig_hist = plot_training_history(history, title=f'Training History - {model_name}')
        fig_hist.savefig(f"../plots/{model_name}_training_history.png")
        plt.close(fig_hist)
    
    if final_test_targets.size > 0 and final_test_preds.size > 0:
        binary_test_preds = (final_test_preds > 0.5).astype(int)
        fig_cm = plot_confusion_matrix(final_test_targets, binary_test_preds, title=f'Confusion Matrix (Test) - {model_name}')
        fig_cm.savefig(f"../plots/{model_name}_confusion_matrix.png")
        plt.close(fig_cm)
        
        fig_roc = plot_roc_curve(final_test_targets, final_test_preds, title=f'ROC Curve (Test) - {model_name}')
        fig_roc.savefig(f"../plots/{model_name}_roc_curve.png")
        plt.close(fig_roc)
    
    plt.close('all') # Close any other stray plots
    
    # Return test metrics if available, otherwise validation, otherwise training, otherwise empty
    if final_test_metrics:
        return final_test_metrics, model
    elif final_val_metrics:
        return final_val_metrics, model
    elif final_train_metrics:
        return final_train_metrics, model
    else:
        return {"loss": float('nan'), "auc": float('nan')}, model # Default if no evaluation happened
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
        'name': 'class_imbalance_focused',  # New configuration
        'hidden_channels': 64,
        'num_layers': 2,
        'num_timesteps': 3,
        'dropout': 0.3,  # Higher dropout to prevent overfitting
        'learning_rate': 0.0005,  # Lower learning rate
        'weight_decay': 1e-4,  # Higher weight decay
        'batch_size': 16,  # Smaller batch size
        'epochs': 75,  # More epochs
        'patience': 15  # More patience
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
    os.makedirs("../../models/classification", exist_ok=True)
    os.makedirs("../../plots", exist_ok=True)

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
                best_model_path = f"../models/classification/{model_name}_attentivefp_best.pt"
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
                with open(f"../models/classification/{model_name}_best_model_info.csv", 'w') as f:
                    f.write("=== Best Model ===\n")
                    best_model_df.to_csv(f, index=False)

    # After processing all datasets, create an overall summary
    if all_results:
        # Create a DataFrame with all results
        all_results_df = pd.DataFrame(all_results)

        # Save all results to CSV
        all_results_df.to_csv("../models/classification/classification_all_results.csv", index=False)

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
        best_models_df.to_csv("../models/classification/classification_best_models_summary.csv", index=False)

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
