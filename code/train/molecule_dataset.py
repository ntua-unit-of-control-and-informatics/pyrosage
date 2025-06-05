import torch
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data

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

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
