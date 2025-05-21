# Pyrosage - Environmental Property Predictors

This repository contains datasets, models, and reproducible notebooks for predicting key physicochemical, environmental, and toxicity-related properties of chemical compounds. These predictors are intended to support virtual screening and safer chemical design, especially for applications like flame retardancy and environmental safety.

## ✅ Covered Properties

The following properties are included and modeled using open datasets:

- LogKow / LogP
- Bioaccumulation Factor (BCF)
- Soil/Water Partition Coefficient (Koc)
- Biodegradability
- Water Solubility
- Henry's Law Constant (KH)
- Aqueous Hydroxyl Rate (kAOH)
- Hydrolytic Stability
- pKa (acidic and basic)
- KOA (Octanol-Air Partition Coefficient)
- TBP (related to biodegradation)
- Fish Bioaccumulation / Toxicity
- Molecular Weight (computed)

## 📂 Repository Structure

```
/data              # CSV datasets used for training
/models            # Saved model files (e.g., .pkl or .onnx)
/notebooks         # Jupyter notebooks to train or evaluate models
README.md          # You're here
```

## 📘 Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/environmental-predictors.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Explore and run any notebook in `/notebooks` to reproduce or fine-tune the models.

## ⚖️ License

This project uses publicly available data (e.g. from VEGA/OPERA repositories) and models trained for research purposes. Please cite sources if you reuse the datasets.

## Acknowledgments

Some of the datasets and models in this repository are reused or adapted from the work of Wang et al. (2023), titled *"Applicability Domains Based on Molecular Graph Contrastive Learning Enable Graph Attention Network Models to Accurately Predict 15 Environmental End Points"*. This study introduced GAT-based QSAR models with applicability domain refinements for environmental property prediction.

📄 Reference: [https://doi.org/10.1021/acs.est.3c03860](https://doi.org/10.1021/acs.est.3c03860)

