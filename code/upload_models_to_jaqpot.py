import os
import torch
from torch_geometric.nn import AttentiveFP
from torch_geometric.data import Data
from rdkit import Chem
from jaqpot_api_client import ModelTask, Feature, FeatureType, ModelVisibility
from jaqpotpy.models.torch_models.torch_onnx import TorchONNXModel
from jaqpotpy import Jaqpot
import pandas as pd


class SMILESPreprocessor:
    """
    Preprocessor for converting SMILES strings to PyTorch Geometric Data objects
    Compatible with AttentiveFP models trained in this project
    """
    
    def __init__(self):
        self.name = "SMILES_to_PyG_Preprocessor"
    
    def preprocess(self, smiles_string):
        """
        Convert SMILES string to PyG Data object with enhanced features
        (matches MoleculeDataset processing from this project)
        
        Parameters:
        -----------
        smiles_string: str
            Input SMILES string
            
        Returns:
        --------
        torch.Tensor: Flattened molecular representation for ONNX compatibility
        """
        data = self.smiles_to_data(smiles_string)
        if data is None:
            # Return zero tensor if molecule parsing fails
            return torch.zeros(1, 100, dtype=torch.float32)  # Placeholder size
        
        # For ONNX compatibility, we need to flatten the graph representation
        # This is a simplified approach - in practice you might need more sophisticated handling
        x_flat = data.x.flatten()
        edge_attr_flat = data.edge_attr.flatten() if data.edge_attr.numel() > 0 else torch.tensor([])
        
        # Pad or truncate to fixed size for ONNX
        max_size = 100
        combined = torch.cat([x_flat, edge_attr_flat])
        
        if combined.size(0) > max_size:
            combined = combined[:max_size]
        else:
            padding = torch.zeros(max_size - combined.size(0))
            combined = torch.cat([combined, padding])
            
        return combined.unsqueeze(0)  # Add batch dimension
    
    def smiles_to_data(self, smiles, target=0.0):
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
    
    def export_to_onnx(self):
        """
        Export preprocessor to ONNX format for Jaqpot compatibility
        For now, returns None as graph preprocessing is complex for ONNX
        """
        print("Warning: ONNX export for graph preprocessing is not implemented.")
        print("Using direct PyTorch preprocessing instead.")
        return None


def create_dummy_molecular_graph(batch_size=1, num_nodes=10):
    """
    Create dummy molecular graph data for ONNX tracing
    This represents the expected input format for AttentiveFP
    
    Parameters:
    -----------
    batch_size: int
        Number of molecules in the batch
    num_nodes: int
        Number of atoms per molecule (will be padded/truncated)
        
    Returns:
    --------
    tuple: (x, edge_index, edge_attr, batch)
        Graph data in AttentiveFP expected format
    """
    # Node features: [total_nodes, 10] (atom features)
    total_nodes = batch_size * num_nodes
    x = torch.randn(total_nodes, 10, dtype=torch.float32)
    
    # Edge indices: create a simple ring structure for each molecule
    edge_indices = []
    for mol_idx in range(batch_size):
        offset = mol_idx * num_nodes
        # Create edges for a ring structure
        for i in range(num_nodes):
            next_node = (i + 1) % num_nodes
            edge_indices.extend([
                [offset + i, offset + next_node],
                [offset + next_node, offset + i]  # bidirectional
            ])
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
    
    # Edge attributes: [num_edges, 6] (bond features)
    num_edges = edge_index.size(1)
    edge_attr = torch.randn(num_edges, 6, dtype=torch.float32)
    
    # Batch tensor: indicates which molecule each node belongs to
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes)
    
    return x, edge_index, edge_attr, batch


def load_trained_model(model_path, device='cpu'):
    """
    Load a trained AttentiveFP model from the project's saved models
    
    Parameters:
    -----------
    model_path: str
        Path to the saved model file
    device: str
        Device to load the model on
        
    Returns:
    --------
    model: AttentiveFP
        The loaded AttentiveFP model
    hyperparams: dict
        Model hyperparameters
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
    return model, hyperparams


def upload_model_to_jaqpot(model_path, model_name, description, model_task_type, visibility=ModelVisibility.PUBLIC):
    """
    Upload an AttentiveFP model to Jaqpot platform using direct ONNX export
    
    Parameters:
    -----------
    model_path: str
        Path to the trained model file
    model_name: str
        Name for the model on Jaqpot
    description: str
        Description of the model
    model_task_type: ModelTask
        Type of task (CLASSIFICATION or REGRESSION)
    visibility: ModelVisibility
        Model visibility setting
        
    Returns:
    --------
    result: dict
        Upload result information
    """
    
    # Load the trained AttentiveFP model
    device = 'cpu'  # Use CPU for ONNX export
    attentivefp_model, hyperparams = load_trained_model(model_path, device)
    
    # Create dummy molecular graph inputs for ONNX tracing
    # This represents the actual inputs that AttentiveFP expects
    x, edge_index, edge_attr, batch = create_dummy_molecular_graph(batch_size=1, num_nodes=10)
    
    # Define features - for graph models, we need to represent the graph structure
    independent_features = [
        Feature(key="x", name="Node Features", feature_type=FeatureType.FLOAT),
        Feature(key="edge_index", name="Edge Indices", feature_type=FeatureType.INTEGER),
        Feature(key="edge_attr", name="Edge Attributes", feature_type=FeatureType.FLOAT),
        Feature(key="batch", name="Batch Index", feature_type=FeatureType.INTEGER)
    ]
    
    if model_task_type == ModelTask.BINARY_CLASSIFICATION:
        dependent_features = [
            Feature(key="prediction", name="Class Prediction", feature_type=FeatureType.FLOAT)
        ]
    else:  # REGRESSION
        dependent_features = [
            Feature(key="prediction", name="Predicted Value", feature_type=FeatureType.FLOAT)
        ]
    
    try:
        # Test if AttentiveFP can be exported to ONNX directly
        print(f"Attempting ONNX export for AttentiveFP model...")
        
        # Try to export the model to ONNX
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_file:
            onnx_path = tmp_file.name
        
        # Export to ONNX using torch.onnx.export
        torch.onnx.export(
            attentivefp_model,
            (x, edge_index, edge_attr, batch),  # Model inputs
            onnx_path,
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            input_names=['node_features', 'edge_indices', 'edge_attributes', 'batch_index'],
            output_names=['prediction'],
            dynamic_axes={
                'node_features': {0: 'num_nodes'},
                'edge_indices': {1: 'num_edges'}, 
                'edge_attributes': {0: 'num_edges'},
                'batch_index': {0: 'num_nodes'}
            }
        )
        
        print(f"ONNX export successful! Model saved to: {onnx_path}")
        
        # Create Jaqpot model using the AttentiveFP model directly
        jaqpot_model = TorchONNXModel(
            attentivefp_model,
            (x, edge_index, edge_attr, batch),  # Use graph inputs
            model_task_type,
            independent_features=independent_features,
            dependent_features=dependent_features,
            onnx_preprocessor=None,  # No preprocessing needed for direct graph input
        )
        
        # Initialize Jaqpot client
        jaqpot = Jaqpot()
        jaqpot.login()
        
        # Deploy model
        result = jaqpot_model.deploy_on_jaqpot(
            jaqpot,
            name=model_name,
            description=description,
            visibility=visibility,
        )
        
        print(f"Successfully uploaded model: {model_name}")
        
        # Clean up temporary ONNX file
        os.unlink(onnx_path)
        
        return result
        
    except Exception as e:
        print(f"Error uploading model {model_name}: {str(e)}")
        print(f"This might be due to PyTorch Geometric operations not being fully supported in ONNX export.")
        print(f"Consider using a simpler model architecture or custom ONNX implementation.")
        return None


def upload_all_classification_models(models_dir="../models/classification"):
    """
    Upload all classification models to Jaqpot
    """
    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('_best.pt')]
    
    results = {}
    for model_file in model_files:
        endpoint_name = model_file.replace('_attentivefp_best.pt', '')
        model_path = os.path.join(models_dir, model_file)
        
        model_name = f"Pyrosage_{endpoint_name}_AttentiveFP"
        description = f"AttentiveFP model for {endpoint_name} classification trained on molecular SMILES data"
        
        print(f"\nUploading classification model: {endpoint_name}")
        result = upload_model_to_jaqpot(
            model_path=model_path,
            model_name=model_name,
            description=description,
            model_task_type=ModelTask.BINARY_CLASSIFICATION,
            visibility=ModelVisibility.PUBLIC
        )
        
        results[endpoint_name] = result
    
    return results


def upload_all_regression_models(models_dir="../models/regression"):
    """
    Upload all regression models to Jaqpot
    """
    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('_best.pt')]
    
    results = {}
    for model_file in model_files:
        endpoint_name = model_file.replace('_attentivefp_best.pt', '')
        model_path = os.path.join(models_dir, model_file)
        
        model_name = f"Pyrosage_{endpoint_name}_AttentiveFP"
        description = f"AttentiveFP model for {endpoint_name} regression trained on molecular SMILES data"
        
        print(f"\nUploading regression model: {endpoint_name}")
        result = upload_model_to_jaqpot(
            model_path=model_path,
            model_name=model_name,
            description=description,
            model_task_type=ModelTask.REGRESSION,
            visibility=ModelVisibility.PUBLIC
        )
        
        results[endpoint_name] = result
    
    return results


def upload_single_model(model_path, endpoint_name, model_type="classification"):
    """
    Upload a single specific model to Jaqpot
    
    Parameters:
    -----------
    model_path: str
        Path to the model file
    endpoint_name: str
        Name of the endpoint (e.g., "AMES", "LC50")
    model_type: str
        "classification" or "regression"
    """
    model_name = f"Pyrosage_{endpoint_name}_AttentiveFP"
    description = f"AttentiveFP model for {endpoint_name} {model_type} trained on molecular SMILES data"
    
    task_type = ModelTask.BINARY_CLASSIFICATION if model_type == "classification" else ModelTask.REGRESSION
    
    print(f"Uploading {model_type} model: {endpoint_name}")
    result = upload_model_to_jaqpot(
        model_path=model_path,
        model_name=model_name,
        description=description,
        model_task_type=task_type,
        visibility=ModelVisibility.PUBLIC
    )
    
    return result


if __name__ == "__main__":
    print("Pyrosage AttentiveFP Models - Jaqpot Upload Script")
    print("=" * 60)
    
    # Example: Upload a single model
    print("\nExample: Uploading AMES classification model...")
    
    ames_model_path = "../models/classification/AMES_attentivefp_best.pt"
    if os.path.exists(ames_model_path):
        result = upload_single_model(
            model_path=ames_model_path,
            endpoint_name="AMES",
            model_type="classification"
        )
        if result:
            print(f"Upload successful! Model ID: {result}")
    else:
        print(f"Model not found: {ames_model_path}")
    
    # Uncomment below to upload all models
    """
    print("\nUploading all classification models...")
    classification_results = upload_all_classification_models()
    
    print("\nUploading all regression models...")
    regression_results = upload_all_regression_models()
    
    print("\nUpload Summary:")
    print(f"Classification models uploaded: {len([r for r in classification_results.values() if r is not None])}")
    print(f"Regression models uploaded: {len([r for r in regression_results.values() if r is not None])}")
    """
