import os
import torch
from torch_geometric.nn import AttentiveFP
from torch_geometric.data import Data
from rdkit import Chem


def load_model(model_path, device='cpu'):
    """
    Load a saved AttentiveFP model from path

    Parameters:
    -----------
    model_path: str
        Path to the saved model
    device: str
        Device to load the model on ('cpu' or 'cuda')

    Returns:
    --------
    model: AttentiveFP
        The loaded model
    """
    model_dict = torch.load(model_path, map_location=torch.device(device))
    state_dict = model_dict['model_state_dict']
    hyperparams = model_dict['hyperparameters']

    # Create model with correct feature dimensions (matching MoleculeDataset)
    model = AttentiveFP(
        in_channels=10,  # Enhanced atom features (10 dimensions)
        hidden_channels=hyperparams["hidden_channels"],
        out_channels=1,
        edge_dim=6,  # Enhanced bond features (6 dimensions)
        num_layers=hyperparams["num_layers"],
        num_timesteps=hyperparams["num_timesteps"],
        dropout=hyperparams["dropout"],
    ).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def smiles_to_data(smiles, target=0.0):
    """
    Convert SMILES string to PyG Data object with enhanced features
    (matches MoleculeDataset processing)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Enhanced atom features (10 dimensions - matching MoleculeDataset)
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
            # Hybridization as one-hot (3 dimensions)
            int(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP),
            int(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2),
            int(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3)
        ]
        atom_features.append(features)

    x = torch.tensor(atom_features, dtype=torch.float)

    # Enhanced bond features (6 dimensions - matching MoleculeDataset)
    edges_list = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges_list.extend([[i, j], [j, i]])

        features = [
            # Bond type as one-hot (4 dimensions)
            int(bond.GetBondType() == Chem.rdchem.BondType.SINGLE),
            int(bond.GetBondType() == Chem.rdchem.BondType.DOUBLE),
            int(bond.GetBondType() == Chem.rdchem.BondType.TRIPLE),
            int(bond.GetBondType() == Chem.rdchem.BondType.AROMATIC),
            # Additional features (2 dimensions)
            int(bond.GetIsConjugated()),
            int(bond.IsInRing())
        ]
        edge_features.extend([features, features])

    if not edges_list:  # Skip molecules with no bonds
        return None

    edge_index = torch.tensor(edges_list, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    y = torch.tensor([target], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def predict(model, input_smiles, input_device='cpu'):
    """
    Make prediction for a single SMILES string using a trained model
    """
    data = smiles_to_data(input_smiles)
    if data is None:
        return None
    data = data.to(input_device)
    batch = torch.zeros(data.num_nodes, dtype=torch.long).to(input_device)
    with torch.no_grad():
        output = model(data.x, data.edge_index, data.edge_attr, batch)
        return output.item()


def load_all_models(models_dirs=None, input_device='cpu'):
    """
    Load all available AttentiveFP models from classification and regression directories

    Parameters:
    -----------
    models_dirs: list or None
        List of model directories to search. If None, uses default paths.
    input_device: str
        Device to load the models on

    Returns:
    --------
    loaded_models: dict
        Dictionary of loaded models categorized by type
    """
    if models_dirs is None:
        models_dirs = ['../models/classification', '../models/regression']
    
    loaded_models = {'classification': {}, 'regression': {}}

    for models_dir in models_dirs:
        if not os.path.exists(models_dir):
            print(f"Directory not found: {models_dir}")
            continue
            
        model_type = 'classification' if 'classification' in models_dir else 'regression'
        model_files = [f for f in os.listdir(models_dir) if f.endswith('_best.pt')]

        for model_file in model_files:
            endpoint_name = model_file.replace('_attentivefp_best.pt', '')
            model_path = os.path.join(models_dir, model_file)

            try:
                model = load_model(model_path, device=input_device)
                loaded_models[model_type][endpoint_name] = {
                    'model': model,
                    'path': model_path,
                    'type': model_type
                }
                print(f"Loaded {model_type} model: {endpoint_name}")
            except Exception as error:
                print(f"Error loading {endpoint_name}: {error}")

    return loaded_models


def predict_single_smiles(input_smiles, models=None, input_device='cpu'):
    """
    Make predictions for a single SMILES string across all loaded models
    
    Parameters:
    -----------
    input_smiles: str
        SMILES string to predict
    models: dict or None
        Pre-loaded models dictionary. If None, loads all models.
    input_device: str
        Device to use for computation
        
    Returns:
    --------
    predictions: dict
        Dictionary with model predictions categorized by type
    """
    if models is None:
        models = load_all_models(input_device=input_device)
    
    predictions = {'classification': {}, 'regression': {}}
    
    for model_type in ['classification', 'regression']:
        for endpoint_name, model_data in models[model_type].items():
            try:
                pred = predict(model_data['model'], input_smiles, input_device)
                predictions[model_type][endpoint_name] = pred
            except Exception as error:
                print(f"Error predicting {endpoint_name} for {input_smiles}: {error}")
                predictions[model_type][endpoint_name] = None
                
    return predictions


def predict_batch_smiles(smiles_list, models=None, input_device='cpu'):
    """
    Make predictions for multiple SMILES strings
    
    Parameters:
    -----------
    smiles_list: list
        List of SMILES strings to predict
    models: dict or None
        Pre-loaded models dictionary. If None, loads all models.
    input_device: str
        Device to use for computation
        
    Returns:
    --------
    all_predictions: dict
        Dictionary with predictions for each SMILES
    """
    if models is None:
        models = load_all_models(input_device=input_device)
    
    all_predictions = {}
    
    for smiles_string in smiles_list:
        print(f"Predicting for: {smiles_string}")
        all_predictions[smiles_string] = predict_single_smiles(smiles_string, models, input_device)
    
    return all_predictions


if __name__ == "__main__":
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load all available models
    print("Loading all trained models...")
    all_models = load_all_models(input_device=device)
    
    total_models = sum(len(models) for models in all_models.values())
    print(f"\nLoaded {total_models} models total:")
    print(f"- Classification models: {len(all_models['classification'])}")
    print(f"- Regression models: {len(all_models['regression'])}")

    # Example SMILES for testing
    test_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "C1=CC=C(C=C1)C(=O)O",       # Benzoic acid
        "CCO"                         # Ethanol
    ]

    print(f"\nMaking predictions for {len(test_smiles)} example compounds...")
    
    # Make predictions for all test SMILES
    results = predict_batch_smiles(test_smiles, all_models, device)
    
    # Display results in a formatted way
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    
    for smiles_string, predictions in results.items():
        print(f"\nSMILES: {smiles_string}")
        print("-" * 60)
        
        print("Classification predictions:")
        for endpoint, value in predictions['classification'].items():
            if value is not None:
                print(f"  {endpoint}: {value:.4f}")
            else:
                print(f"  {endpoint}: Error")
        
        print("Regression predictions:")
        for endpoint, value in predictions['regression'].items():
            if value is not None:
                print(f"  {endpoint}: {value:.4f}")
            else:
                print(f"  {endpoint}: Error")
