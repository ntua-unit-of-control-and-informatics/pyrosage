
import os
from os.path import isfile, join
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from rdkit import RDLogger
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentiveFP

RDLogger.DisableLog('rdApp.*')

# === Settings ===
DATA_DIR = "../data/regression"
MODEL_DIR = "../models"
BATCH_SIZE = 32
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MoleculeDataset(torch.utils.data.Dataset):
    def __init__(self, df, smiles_col='smiles', target_col='active'):
        self.data_list = []
        for _, row in df.iterrows():
            mol = Chem.MolFromSmiles(row[smiles_col])
            if mol is None:
                continue
            atom_features = []
            for atom in mol.GetAtoms():
                atom_features.append([
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetFormalCharge(),
                    atom.GetNumRadicalElectrons(),
                    int(atom.GetHybridization()),
                    int(atom.GetIsAromatic()),
                    atom.GetTotalNumHs()
                ])
            x = torch.tensor(atom_features, dtype=torch.float)

            edge_index = []
            edge_attr = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                features = [
                    bond.GetBondTypeAsDouble(),
                    float(bond.GetIsConjugated()),
                    float(bond.IsInRing())
                ]
                edge_index += [[i, j], [j, i]]
                edge_attr += [features, features]

            if not edge_index:
                continue

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            y = torch.tensor([row[target_col]], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def train_model(model_name):
    print(f"\n=== Training {model_name} ===")
    path = os.path.join(DATA_DIR, f"{model_name}.csv")
    df = pd.read_csv(path)

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # Datasets and loaders
    train_loader = DataLoader(MoleculeDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MoleculeDataset(val_df), batch_size=BATCH_SIZE)
    test_loader = DataLoader(MoleculeDataset(test_df), batch_size=BATCH_SIZE)

    # Model
    model = AttentiveFP(
        in_channels=7,
        edge_dim=3,
        hidden_channels=64,
        out_channels=1,
        num_layers=2,
        num_timesteps=2,
        dropout=0.2
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view(-1)
            loss = loss_fn(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view(-1)
                preds.extend(out.cpu().numpy())
                trues.extend(batch.y.cpu().numpy())
        val_loss = mean_squared_error(trues, preds)
        val_r2 = r2_score(trues, preds)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val MSE: {val_loss:.4f} - R²: {val_r2:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"model_{model_name}_geometric.pt"))
            print("✓ Saved best model")

# === Run for all your regression datasets ===
if __name__ == "__main__":
    all_datasets = [f for f in os.listdir(DATA_DIR) if isfile(join(DATA_DIR, f))]
    for dataset_name in all_datasets:
        filename = Path(dataset_name)
        model_name = filename.name.removesuffix("".join(filename.suffixes))

        try:
            train_model(model_name)
        except Exception as e:
            print(f"Failed to train {model_name}: {e}")
