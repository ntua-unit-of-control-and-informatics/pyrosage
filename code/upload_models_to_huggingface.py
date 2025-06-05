import json
import os
import shutil
import tempfile
import time

import pandas as pd
import torch
from huggingface_hub import HfApi, create_repo, upload_folder
from huggingface_hub.utils import HfFolder


class PyrosageModelUploader:
    """
    Upload Pyrosage AttentiveFP models to Hugging Face Hub with proper documentation
    """
    
    def __init__(self, hf_username=None, organization=None):
        """
        Initialize the uploader
        
        Parameters:
        -----------
        hf_username: str
            Hugging Face username. If None, will try to get from token
        organization: str
            Organization name to upload to (e.g., "upci-ntua")
        """
        self.api = HfApi()
        self.hf_username = hf_username or self._get_username()
        self.organization = organization
        
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
    
    def get_endpoint_description(self, endpoint_name, model_type):
        """Get detailed description for each endpoint"""
        
        descriptions = {
            # Classification models
            "AMES": "mutagenicity using the Ames test. Predicts whether a compound can cause mutations in bacterial DNA, which is an important indicator of potential carcinogenicity.",
            "Endocrine_Disruption_NR-AR": "endocrine disruption via androgen receptor (AR). Predicts whether a compound can interfere with androgen hormone signaling, which affects reproductive health.",
            "Endocrine_Disruption_NR-AhR": "endocrine disruption via aryl hydrocarbon receptor (AhR). Predicts whether a compound activates the AhR pathway, which is involved in toxicological responses.",
            "Endocrine_Disruption_NR-ER": "endocrine disruption via estrogen receptor (ER). Predicts whether a compound can bind to and activate estrogen receptors, affecting hormonal balance.",
            "Endocrine_Disruption_NR-aromatase": "endocrine disruption via aromatase inhibition. Predicts whether a compound can inhibit the aromatase enzyme, which converts androgens to estrogens.",
            "Irritation_Corrosion_Eye_Corrosion": "eye corrosion potential. Predicts whether a compound can cause irreversible damage to eye tissue upon contact.",
            "Irritation_Corrosion_Eye_Irritation": "eye irritation potential. Predicts whether a compound can cause reversible inflammatory responses in eye tissue.",
            
            # Regression models - Environmental & Physicochemical Properties
            "KOA": "the octanol-air partition coefficient (log KOA). This property measures a compound's tendency to partition between octanol and air, indicating volatility and environmental transport potential.",
            "KOC": "the organic carbon partition coefficient (log KOC). This property predicts soil adsorption behavior and is key for environmental mobility assessment.",
            "KOW": "the octanol-water partition coefficient (log KOW/log P). This fundamental property indicates lipophilicity and affects bioaccumulation potential.",
            "SW": "aqueous solubility (log SW). This property affects environmental fate, bioavailability, and exposure potential.",
            "KH": "Henry's Law constant (log KH). This property relates water-air partitioning and is crucial for modeling volatilization from water bodies.",
            "kAOH": "the reaction rate constant with hydroxyl radicals (log kAOH). This property relates to atmospheric degradation and environmental persistence.",
            "FBA": "fugacity-based environmental fate parameter A. This property is used in fugacity models for environmental distribution prediction.",
            "FBC": "fugacity-based environmental fate parameter C. This property is used in fugacity models for environmental distribution prediction.",
            
            # Regression models - Toxicity & Bioactivity
            "LC50": "aquatic toxicity (log LC50). This property predicts the lethal concentration for 50% of aquatic organisms (fish, daphnia), crucial for ecological risk assessment.",
            "LD50_Zhu": "acute oral toxicity (log LD50). This property predicts the lethal dose for 50% of test animals, important for mammalian toxicity assessment.",
            "tbiodeg": "biodegradability potential. This property classifies whether a compound is readily biodegradable in the environment.",
            "TBP": "teratogenic bioaccumulation potential. This property may relate to developmental toxicity or bioaccumulation scoring.",
            "tfishbio": "fish bioaccumulation factor. This property measures accumulation in fish tissue, important for food chain modeling and ecological risk.",
            "TMP": "toxicity-related molecular property. This endpoint represents an experimental toxicity measurement.",
            
            # Regression models - Acid/Base Behavior
            "pKa_acidic": "the acid dissociation constant (pKa) for acidic groups. This property predicts the pH at which acidic functional groups donate protons, affecting ionization state and bioavailability.",
            "pKa_basic": "the acid dissociation constant (pKa) for basic groups. This property predicts the pH at which basic functional groups accept protons, important for ADME properties.",
            
            # Others
            "PLV": "plasma level volume or protein binding parameter. This property may relate to pharmacokinetics or blood partitioning behavior."
        }
        
        return descriptions.get(endpoint_name, f"{endpoint_name} molecular property")
    
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
        endpoint_description = self.get_endpoint_description(endpoint_name, model_type)
        
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

This is an AttentiveFP (Attention-based Fingerprint) Graph Neural Network model trained to predict {endpoint_description} The model takes SMILES strings as input and uses graph neural networks to predict molecular properties directly from the molecular structure.

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
  author={{UPCI NTUA}},
  year={{2025}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/{self.organization or self.hf_username or "USER"}/pyrosage-{endpoint_name.lower()}-attentivefp}}
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
    
    def upload_model(self, model_path, endpoint_name, model_type, repo_name=None, private=False, max_retries=3):
        """
        Upload a single model to Hugging Face Hub with retry logic
        
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
        max_retries: int
            Maximum number of retry attempts for rate limits
            
        Returns:
        --------
        str: Repository URL if successful, None otherwise
        """
        
        if not self.hf_username:
            print("Error: No Hugging Face username found. Please login with `huggingface-cli login`")
            return None
        
        if repo_name is None:
            repo_name = f"pyrosage-{endpoint_name.lower()}-attentivefp"
        
        # Use organization if specified, otherwise use username
        namespace = self.organization if self.organization else self.hf_username
        repo_id = f"{namespace}/{repo_name}"
        
        for attempt in range(max_retries + 1):
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
                error_str = str(e)
                if "429" in error_str or "Too Many Requests" in error_str:
                    if attempt < max_retries:
                        wait_time = (attempt + 1) * 30  # Progressive delay: 30s, 60s, 90s
                        print(f"⏳ Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"❌ Rate limit exceeded after {max_retries} retries for {endpoint_name}")
                        return None
                else:
                    print(f"❌ Error uploading {endpoint_name}: {error_str}")
                    return None
        
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
    
    def upload_missing_models(self, missing_models, model_type, models_dir, private=False):
        """
        Upload only specific missing models
        
        Parameters:
        -----------
        missing_models: list
            List of model endpoint names that failed to upload
        model_type: str
            "classification" or "regression"
        models_dir: str
            Directory containing the model files
        private: bool
            Whether to make repositories private
            
        Returns:
        --------
        dict: Results of upload attempts
        """
        
        if not os.path.exists(models_dir):
            print(f"Models directory not found: {models_dir}")
            return {}
        
        results = {}
        
        print(f"🔄 Retrying upload for {len(missing_models)} missing {model_type} models...")
        
        for endpoint_name in missing_models:
            model_file = f"{endpoint_name}_attentivefp_best.pt"
            model_path = os.path.join(models_dir, model_file)
            
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                results[endpoint_name] = None
                continue
            
            print(f"\n🔄 Retrying {model_type} model: {endpoint_name}")
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
    
    # Initialize uploader with UPCI-NTUA organization
    uploader = PyrosageModelUploader(organization="upci-ntua")
    
    if not uploader.hf_username:
        print("❌ Please login to Hugging Face first:")
        print("   pip install huggingface_hub")
        print("   huggingface-cli login")
        return
    
    print(f"📝 Uploading models to organization: upci-ntua")
    print(f"👤 Using credentials from user: {uploader.hf_username}")
    
    # Example: Upload a single model
    # print("\\n🧪 Example: Uploading AMES classification model...")
    # ames_model_path = "../models/classification/AMES_attentivefp_best.pt"
    # if os.path.exists(ames_model_path):
    #     repo_url = uploader.upload_model(
    #         model_path=ames_model_path,
    #         endpoint_name="AMES",
    #         model_type="classification"
    #     )
    #     if repo_url:
    #         print(f"🎉 AMES model uploaded: {repo_url}")
    # else:
    #     print(f"❌ Model not found: {ames_model_path}")
    
    # Upload all models with updated descriptions
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
        else:
            print(f"   {endpoint}: ❌ Failed to upload")


if __name__ == "__main__":
    main()
