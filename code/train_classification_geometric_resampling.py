# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Molecular Property Prediction with AttentiveFP
# Using PyTorch Geometric's implementation of AttentiveFP for molecular property prediction.
#

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentiveFP
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from rdkit import RDLogger
import numpy as np
from sklearn.preprocessing import StandardScaler

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

def generate_morgan_fingerprints(smiles):
    """Generate Morgan fingerprints for SMOTE processing"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))



# %% [markdown]
# # Data Preparation
# Create a PyTorch Geometric dataset from SMILES strings.
#

# %%
class MoleculeDataset(Dataset):
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
                
            # Get node features
            num_atoms = mol.GetNumAtoms()
            atom_features = []
            for atom in mol.GetAtoms():
                atom_features.append([
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetFormalCharge(),
                    atom.GetNumRadicalElectrons(),
                    atom.GetHybridization(),
                    atom.GetIsAromatic(),
                    atom.GetTotalNumHs()
                ])
            x = torch.tensor(atom_features, dtype=torch.float)
            
            # Get edge indices and features
            edges_list = []
            edge_features = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edges_list.extend([[i, j], [j, i]])
                
                # Bond features
                bond_type = bond.GetBondType()
                features = [
                    bond_type == Chem.rdchem.BondType.SINGLE,
                    bond_type == Chem.rdchem.BondType.DOUBLE,
                    bond_type == Chem.rdchem.BondType.TRIPLE,
                    bond_type == Chem.rdchem.BondType.AROMATIC,
                    bond.GetIsConjugated(),
                ]
                edge_features.extend([features, features])
                
            edge_index = torch.tensor(edges_list, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            
            # Create PyG Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([row[self.target_col]], dtype=torch.float),
            )
            data_list.append(data)
        return data_list
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]



# %% jupyter={"is_executing": true}
# Load data
print("Loading and preparing data...")
model_name = 'Endocrine_Disruption_NR-aromatase'
df = pd.read_csv(f"../data/{model_name}.csv")

# Generate fingerprints for SMOTE
print("Generating molecular fingerprints...")
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

# Print class distribution before SMOTE
print("\nClass distribution before SMOTE:")
print(pd.Series(y).value_counts(normalize=True))

# Apply SMOTE
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Create new balanced dataframe
print("\nCreating balanced dataset...")
balanced_smiles = []
original_smiles = df['smiles'].iloc[valid_indices].values

# Map original samples
original_indices = range(len(y))
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

# Print final class distribution
print("\nClass distribution after SMOTE:")
print(balanced_df['active'].value_counts(normalize=True))

# Split balanced data
print("\nSplitting data...")
train_df, test_df = train_test_split(balanced_df, test_size=0.2, random_state=42, stratify=balanced_df['active'])
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['active'])

print("\nFinal dataset sizes:")
print(f"Training set: {len(train_df)}")
print(f"Validation set: {len(val_df)}")
print(f"Test set: {len(test_df)}")

# Create datasets using the balanced data
train_dataset = MoleculeDataset(train_df)
val_dataset = MoleculeDataset(val_df)
test_dataset = MoleculeDataset(test_df)

# Create data loaders (no need for weighted sampling now as dataset is balanced)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# Load data
#model_name = 'Endocrine_Disruption_NR-ER'
#df = pd.read_csv(f"../data/{model_name}.csv")

# Split data
#train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['active'])
#train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['active'])

# Create datasets
#train_dataset = MoleculeDataset(train_df)
#val_dataset = MoleculeDataset(val_df)
#test_dataset = MoleculeDataset(test_df)

# Create data loaders
#train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
#test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")


# %% [markdown]
# # Model Definition
# Using PyTorch Geometric's AttentiveFP implementation
#

# %%
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model
model = AttentiveFP(
    in_channels=7,  # Number of atom features
    hidden_channels=64,
    out_channels=1,  # Binary classification
    edge_dim=5,  # Number of bond features
    num_layers=3,
    num_timesteps=2,
    dropout=0.1
).to(device)

# Loss function with class weights
pos_weight = torch.tensor([
    (len(train_df) - train_df['active'].sum()) / train_df['active'].sum()
]).to(device)
criterion = nn.BCEWithLogitsLoss()  # (pos_weight=pos_weight)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)


# %% [markdown]
# # Training and Evaluation Functions
#

# %%
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        
        target = batch.y
        
        loss = criterion(out.squeeze(-1), target)  # Now shapes will match
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        predictions.extend(out.detach().cpu().numpy())
        targets.extend(batch.y.cpu().numpy())
    
    return (
        total_loss / len(loader.dataset),
        np.array(predictions),
        np.array(targets)
    )

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(
                batch.x, batch.edge_index, batch.edge_attr,
                batch=batch.batch
            )
            loss = criterion(out.squeeze(-1), batch.y)
            
            total_loss += loss.item() * batch.num_graphs
            predictions.extend(torch.sigmoid(out).cpu().numpy())
            targets.extend(batch.y.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    return {
        'loss': total_loss / len(loader.dataset),
        'accuracy': accuracy_score(targets, predictions > 0.5),
        'precision': precision_score(targets, predictions > 0.5),
        'recall': recall_score(targets, predictions > 0.5),
        'f1': f1_score(targets, predictions > 0.5),
        'auc': roc_auc_score(targets, predictions)
    }



# %% [markdown]
# # Training Loop
#

# %%
num_epochs = 50
best_val_loss = float('inf')
patience = 10
patience_counter = 0
history = {
    'train_loss': [], 'val_loss': [],
    'val_accuracy': [], 'val_f1': [], 'val_auc': []
}

for epoch in range(num_epochs):
    # Training
    train_loss, train_preds, train_targets = train_epoch(
        model, train_loader, optimizer, criterion, device
    )
    
    # Validation
    val_metrics = evaluate(model, val_loader, criterion, device)
    
    # Update learning rate
    scheduler.step(val_metrics['loss'])
    
    # Save metrics
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_metrics['loss'])
    history['val_accuracy'].append(val_metrics['accuracy'])
    history['val_f1'].append(val_metrics['f1'])
    history['val_auc'].append(val_metrics['auc'])
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_metrics['loss']:.4f}")
    print(f"Val Metrics: Acc={val_metrics['accuracy']:.3f}, "
          f"F1={val_metrics['f1']:.3f}, AUC={val_metrics['auc']:.3f}")
    
    # Early stopping
    if val_metrics['loss'] < best_val_loss:
        best_val_loss = val_metrics['loss']
        patience_counter = 0
        torch.save(model.state_dict(), f"../models/model_{model_name}_geometric.pt")
        print("✓ Saved best model")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break
    print("-" * 50)


# %% [markdown]
# # Evaluation and Visualization
#

# %%
# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['val_accuracy'], label='Accuracy')
plt.plot(history['val_f1'], label='F1 Score')
plt.plot(history['val_auc'], label='AUC')
plt.title('Validation Metrics')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()

plt.tight_layout()
plt.show()

# Final evaluation on test set
model.load_state_dict(torch.load(f"../models/model_{model_name}_geometric.pt"))
test_metrics = evaluate(model, test_loader, criterion, device)

print("\nTest Set Metrics:")
for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")
