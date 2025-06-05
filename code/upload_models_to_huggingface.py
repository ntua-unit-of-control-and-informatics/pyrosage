import os
import json
import torch
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, Repository, create_repo, upload_folder
from huggingface_hub.utils import HfFolder
import pandas as pd


class PyrosageModelUploader:
    """
    Upload Pyrosage AttentiveFP models to Hugging Face Hub with proper documentation
    """
    
    def __init__(self, hf_username=None):
        """
        Initialize the uploader
        
        Parameters:
        -----------
        hf_username: str
            Hugging Face username. If None, will try to get from token
        """
        self.api = HfApi()
        self.hf_username = hf_username or self._get_username()
        
    def _get_username(self):
        """Get username from HF token"""
        try:
            token = HfFolder.get_token()
            if token:
                user_info = self.api.whoami(token=token)
                return user_info["name"]
        except:
            pass
        return None
    
    def create_model_card(self, endpoint_name, model_type, hyperparams, model_info=None):
        """
        Create a comprehensive model card for the AttentiveFP model
        
        Parameters:
        -----------
        endpoint_name: str
            Name of the endpoint (e.g., "AMES", "LC50")
        model_type: str
            "classification" or "regression"
        hyperparams: dict
            Model hyperparameters
        model_info: dict
            Additional model information (metrics, etc.)
            
        Returns:
        --------
        str: Model card content in markdown format
        """
        
        task_type = "Binary Classification" if model_type == "classification" else "Regression"
        
        model_card = f"""---
license: mit
tags:
- chemistry
- molecular-property-prediction
- graph-neural-networks
- attentivefp
- pytorch-geometric
- toxicity-prediction
language:
- en
pipeline_tag: {"text-classification" if model_type == "classification" else "tabular-regression"}
---

# Pyrosage {endpoint_name} AttentiveFP Model

## Model Description

This is an AttentiveFP (Attention-based Fingerprint) Graph Neural Network model trained for {endpoint_name} {task_type.lower()} from the Pyrosage project. The model predicts molecular properties directly from SMILES strings using graph neural networks.

## Model Details

- **Model Type**: AttentiveFP (Graph Neural Network)
- **Task**: {task_type}
- **Input**: SMILES strings (molecular representations)
- **Output**: {"Binary classification (0/1)" if model_type == "classification" else "Continuous numerical value"}
- **Framework**: PyTorch Geometric
- **Architecture**: AttentiveFP with enhanced atom and bond features

### Hyperparameters

```json
{json.dumps(hyperparams, indent=2)}
```

## Usage

### Installation

```bash
pip install torch torch-geometric rdkit-pypi
```

### Loading the Model

```python
import torch
from torch_geometric.nn import AttentiveFP
from rdkit import Chem
from torch_geometric.data import Data

# Load the model
model_dict = torch.load('pytorch_model.pt', map_location='cpu')
state_dict = model_dict['model_state_dict']
hyperparams = model_dict['hyperparameters']

# Create model with correct architecture
model = AttentiveFP(
    in_channels=10,  # Enhanced atom features
    hidden_channels=hyperparams["hidden_channels"],
    out_channels=1,
    edge_dim=6,  # Enhanced bond features
    num_layers=hyperparams["num_layers"],
    num_timesteps=hyperparams["num_timesteps"],
    dropout=hyperparams["dropout"],
)

model.load_state_dict(state_dict)
model.eval()
```

### Making Predictions

```python
def smiles_to_data(smiles):
    \"\"\"Convert SMILES string to PyG Data object\"\"\"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Enhanced atom features (10 dimensions)
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

    # Enhanced bond features (6 dimensions)
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

    if not edges_list:
        return None

    edge_index = torch.tensor(edges_list, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def predict(model, smiles):
    \"\"\"Make prediction for a SMILES string\"\"\"
    data = smiles_to_data(smiles)
    if data is None:
        return None
    
    batch = torch.zeros(data.num_nodes, dtype=torch.long)
    with torch.no_grad():
        output = model(data.x, data.edge_index, data.edge_attr, batch)
        return output.item()

# Example usage
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
prediction = predict(model, smiles)
print(f"Prediction for {{smiles}}: {{prediction}}")
```

## Training Data

The model was trained on the {endpoint_name} dataset from the Pyrosage project, which focuses on molecular toxicity and environmental property prediction.

## Model Performance

{f"Training metrics: {model_info}" if model_info else "See training logs for detailed performance metrics."}

## Limitations

- The model is trained on specific chemical datasets and may not generalize to all molecular types
- Performance may vary for molecules significantly different from the training distribution
- Requires proper SMILES string format for input

## Citation

If you use this model, please cite the Pyrosage project:

```bibtex
@misc{{pyrosage{endpoint_name.lower()},
  title={{Pyrosage {endpoint_name} AttentiveFP Model}},
  author={{Pyrosage Team}},
  year={{2024}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/{self.hf_username or "USER"}/pyrosage-{endpoint_name.lower()}-attentivefp}}
}}
```

## License

MIT License - see LICENSE file for details.
"""
        return model_card
    
    def create_inference_script(self, endpoint_name, model_type):
        """Create a standalone inference script"""
        
        script = f'''#!/usr/bin/env python3
"""
Standalone inference script for Pyrosage {endpoint_name} AttentiveFP Model
Usage: python inference.py "SMILES_STRING"
"""

import sys
import torch
from torch_geometric.nn import AttentiveFP
from rdkit import Chem
from torch_geometric.data import Data


def smiles_to_data(smiles):
    """Convert SMILES string to PyG Data object with enhanced features"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Enhanced atom features (10 dimensions)
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

    # Enhanced bond features (6 dimensions)
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

    if not edges_list:
        return None

    edge_index = torch.tensor(edges_list, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def load_model():
    """Load the AttentiveFP model"""
    model_dict = torch.load('pytorch_model.pt', map_location='cpu')
    state_dict = model_dict['model_state_dict']
    hyperparams = model_dict['hyperparameters']

    model = AttentiveFP(
        in_channels=10,  # Enhanced atom features
        hidden_channels=hyperparams["hidden_channels"],
        out_channels=1,
        edge_dim=6,  # Enhanced bond features
        num_layers=hyperparams["num_layers"],
        num_timesteps=hyperparams["num_timesteps"],
        dropout=hyperparams["dropout"],
    )

    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(model, smiles):
    """Make prediction for a SMILES string"""
    data = smiles_to_data(smiles)
    if data is None:
        return None
    
    batch = torch.zeros(data.num_nodes, dtype=torch.long)
    with torch.no_grad():
        output = model(data.x, data.edge_index, data.edge_attr, batch)
        return output.item()


def main():
    if len(sys.argv) != 2:
        print("Usage: python inference.py 'SMILES_STRING'")
        print("Example: python inference.py 'CC(=O)OC1=CC=CC=C1C(=O)O'")
        sys.exit(1)
    
    smiles = sys.argv[1]
    print(f"Loading {endpoint_name} AttentiveFP model...")
    
    try:
        model = load_model()
        print(f"Making prediction for: {{smiles}}")
        
        prediction = predict(model, smiles)
        if prediction is not None:
            {"print(f'Classification result: {prediction:.4f} (>0.5 = positive, <=0.5 = negative)')" if model_type == "classification" else "print(f'Regression result: {prediction:.4f}')"}
        else:
            print("Error: Could not process SMILES string")
            
    except Exception as e:
        print(f"Error: {{e}}")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''
        return script
    
    def prepare_model_files(self, model_path, endpoint_name, model_type, temp_dir):
        """
        Prepare all files needed for the model repository
        
        Parameters:
        -----------
        model_path: str
            Path to the original model file
        endpoint_name: str
            Name of the endpoint
        model_type: str
            "classification" or "regression"
        temp_dir: str
            Temporary directory to prepare files
            
        Returns:
        --------
        dict: Information about prepared files
        """
        
        # Load model to get hyperparameters
        model_dict = torch.load(model_path, map_location='cpu')
        hyperparams = model_dict['hyperparameters']
        
        # Copy model file (keeping original .pt format but renaming for HF convention)
        model_file = os.path.join(temp_dir, "pytorch_model.pt")
        shutil.copy2(model_path, model_file)
        
        # Load model info if available
        model_info = None
        model_info_path = model_path.replace('_best.pt', '_best_model_info.csv')
        if os.path.exists(model_info_path):
            try:
                df = pd.read_csv(model_info_path)
                model_info = df.to_dict('records')[0] if len(df) > 0 else None
            except:
                pass
        
        # Create model card
        model_card = self.create_model_card(endpoint_name, model_type, hyperparams, model_info)
        with open(os.path.join(temp_dir, "README.md"), "w") as f:
            f.write(model_card)
        
        # Create inference script
        inference_script = self.create_inference_script(endpoint_name, model_type)
        with open(os.path.join(temp_dir, "inference.py"), "w") as f:
            f.write(inference_script)
        
        # Create requirements.txt
        requirements = """torch>=1.9.0
torch-geometric>=2.0.0
rdkit-pypi>=2022.3.0
numpy>=1.21.0
"""
        with open(os.path.join(temp_dir, "requirements.txt"), "w") as f:
            f.write(requirements)
        
        # Create config.json with model metadata
        config = {
            "model_type": "AttentiveFP",
            "task_type": model_type,
            "endpoint": endpoint_name,
            "hyperparameters": hyperparams,
            "input_features": {
                "atom_features": 10,
                "bond_features": 6
            },
            "framework": "pytorch_geometric"
        }
        with open(os.path.join(temp_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        return {
            "hyperparams": hyperparams,
            "model_info": model_info,
            "files_created": ["pytorch_model.pt", "README.md", "inference.py", "requirements.txt", "config.json"]
        }
    
    def upload_model(self, model_path, endpoint_name, model_type, repo_name=None, private=False):
        """
        Upload a single model to Hugging Face Hub
        
        Parameters:
        -----------
        model_path: str
            Path to the model file
        endpoint_name: str
            Name of the endpoint
        model_type: str
            "classification" or "regression"
        repo_name: str
            Custom repository name (optional)
        private: bool
            Whether to make the repository private
            
        Returns:
        --------
        str: Repository URL if successful, None otherwise
        """
        
        if not self.hf_username:
            print("Error: No Hugging Face username found. Please login with `huggingface-cli login`")
            return None
        
        if repo_name is None:
            repo_name = f"pyrosage-{endpoint_name.lower()}-attentivefp"
        
        repo_id = f"{self.hf_username}/{repo_name}"
        
        try:
            # Create repository
            print(f"Creating repository: {repo_id}")
            create_repo(repo_id, private=private, exist_ok=True)
            
            # Prepare files in temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"Preparing model files for {endpoint_name}...")
                file_info = self.prepare_model_files(model_path, endpoint_name, model_type, temp_dir)
                
                print(f"Uploading files to {repo_id}...")
                # Upload the entire folder
                upload_folder(
                    folder_path=temp_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Upload {endpoint_name} AttentiveFP model"
                )
            
            repo_url = f"https://huggingface.co/{repo_id}"
            print(f"✅ Successfully uploaded model: {repo_url}")
            return repo_url
            
        except Exception as e:
            print(f"❌ Error uploading {endpoint_name}: {str(e)}")
            return None
    
    def upload_all_classification_models(self, models_dir="../models/classification", private=False):
        """Upload all classification models"""
        return self._upload_all_models(models_dir, "classification", private)
    
    def upload_all_regression_models(self, models_dir="../models/regression", private=False):
        """Upload all regression models"""
        return self._upload_all_models(models_dir, "regression", private)
    
    def _upload_all_models(self, models_dir, model_type, private):
        """Helper method to upload all models of a given type"""
        if not os.path.exists(models_dir):
            print(f"Models directory not found: {models_dir}")
            return {}
        
        model_files = [f for f in os.listdir(models_dir) if f.endswith('_best.pt')]
        results = {}
        
        print(f"Found {len(model_files)} {model_type} models to upload")
        
        for model_file in model_files:
            endpoint_name = model_file.replace('_attentivefp_best.pt', '')
            model_path = os.path.join(models_dir, model_file)
            
            print(f"\\n📤 Uploading {model_type} model: {endpoint_name}")
            repo_url = self.upload_model(
                model_path=model_path,
                endpoint_name=endpoint_name,
                model_type=model_type,
                private=private
            )
            
            results[endpoint_name] = repo_url
        
        return results


def main():
    print("🚀 Pyrosage Models - Hugging Face Upload Script")
    print("=" * 60)
    
    # Initialize uploader
    uploader = PyrosageModelUploader()
    
    if not uploader.hf_username:
        print("❌ Please login to Hugging Face first:")
        print("   pip install huggingface_hub")
        print("   huggingface-cli login")
        return
    
    print(f"📝 Uploading models for user: {uploader.hf_username}")
    
    # Example: Upload a single model
    print("\\n🧪 Example: Uploading AMES classification model...")
    ames_model_path = "../models/classification/AMES_attentivefp_best.pt"
    if os.path.exists(ames_model_path):
        repo_url = uploader.upload_model(
            model_path=ames_model_path,
            endpoint_name="AMES",
            model_type="classification"
        )
        if repo_url:
            print(f"🎉 AMES model uploaded: {repo_url}")
    else:
        print(f"❌ Model not found: {ames_model_path}")
    
    # Uncomment below to upload all models
    """
    print("\\n📦 Uploading all classification models...")
    classification_results = uploader.upload_all_classification_models()
    
    print("\\n📦 Uploading all regression models...")  
    regression_results = uploader.upload_all_regression_models()
    
    print("\\n📊 Upload Summary:")
    successful_class = len([r for r in classification_results.values() if r is not None])
    successful_reg = len([r for r in regression_results.values() if r is not None])
    
    print(f"   Classification models uploaded: {successful_class}/{len(classification_results)}")
    print(f"   Regression models uploaded: {successful_reg}/{len(regression_results)}")
    
    print("\\n🔗 Repository URLs:")
    for endpoint, url in {**classification_results, **regression_results}.items():
        if url:
            print(f"   {endpoint}: {url}")
    """


if __name__ == "__main__":
    main()