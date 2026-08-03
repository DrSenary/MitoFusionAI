# MitoFusionAI

A machine-learning framework for predicting mitochondrial disruption and prioritizing FDA-approved drug repurposing candidates from high-throughput phenotypic screening data.

## Overview

MitoFusionAI is a six-stage computational pipeline that trains a scaffold-stratified XGBoost classifier on PubChem BioAssay AID 1362 (194,235 compounds screened in a *Saccharomyces cerevisiae* mitochondrial-fusion reporter assay), validates the model end-to-end against an orthogonal confirmatory dose-response screen (AID 1361), and applies the trained model to a curated FDA-approved drug library to identify novel repurposing candidates for neurodegeneration.

### Key Results

| Metric | Value |
|--------|-------|
| Internal test ROC-AUC | 0.849 |
| External validation Spearman ρ (vs. WT IC₅₀) | −0.493 (p < 0.0001) |
| External validation Spearman ρ (vs. selectivity ratio) | +0.451 (p < 0.0001) |
| FDA drugs screened | 1,399 |
| Novel candidates identified | 1,082 |
| Priority CNS-permeable candidates | 20 |
| Top novel lead | Loperamide (P = 0.831) |

## Pipeline Architecture

```
Stage 1: Data Cleaning & Curation
    └── 194,235 compounds → dual-criteria labeling → Bemis-Murcko scaffolds
Stage 2: Baseline Model Training
    └── Scaffold-stratified split → 91-dim features → XGBoost (default params)
Stage 3: Hyperparameter Optimization
    └── Optuna TPE Bayesian search (50 trials, 3-fold CV, PR-AUC objective)
Stage 4: Optimized Model Training
    └── XGBoost with optimized hyperparameters → SHAP interpretability
Stage 5: External Validation
    └── AID 1361 holdout (1,146 compounds) → Spearman correlation vs. IC₅₀
Stage 6: FDA Drug Repurposing
    └── DrugBank library → 5-stage filtering cascade → 20 priority candidates
```

## Repository Structure

```
MitoFusionAI/
│
├── 01_data_cleaning_and_curation.py      # Stage 1: Load AID 1362, clean, label, extract scaffolds
├── 02_baseline_model_training.py          # Stage 2: Scaffold split, featurize, train baseline XGBoost
├── 03_hyperparameter_optimization.py      # Stage 3: Optuna TPE Bayesian search (50 trials)
├── 04_optimized_model_training.py         # Stage 4: Train final model with optimized params + SHAP
├── 05_external_validation.py              # Stage 5: Validate against AID 1361 dose-response data
├── 06_fda_drug_repurposing.py             # Stage 6: Screen FDA drugs, 5-stage filtering cascade
│
├── optuna_best_params.json                # Best hyperparameters from Stage 3
├── requirements.txt                        # Python dependencies
├── README.md                               # This file
│
└── docs/                                   # (Optional) Technical documentation and figures
```

## Requirements

```
Python 3.12+
RDKit 2023.09.5
XGBoost 2.0
scikit-learn 1.4
Optuna 3.5
SHAP 0.44
pandas 2.2
numpy 1.26
matplotlib 3.8
seaborn 0.13
joblib
```

Install with:
```bash
pip install rdkit xgboost scikit-learn optuna shap pandas numpy matplotlib seaborn joblib
```

## Input Data

The following files must be placed in the same directory as the scripts:

| File | Source | Description |
|------|--------|-------------|
| `fusion_inhibition_data.csv` | [PubChem AID 1362](https://pubchem.ncbi.nlm.nih.gov/bioassay/1362) | Primary screen: 194,235 compounds, WT + Mut inhibition |
| `AID_1361_datatable.csv` | [PubChem AID 1361](https://pubchem.ncbi.nlm.nih.gov/bioassay/1361) | Confirmatory dose-response: 1,147 compounds, IC₅₀ values |
| `fda_drugs.sdf` | [DrugBank](https://www.drugbank.com/) | FDA-approved drug structures (SDF format) |

## Usage

Run the scripts sequentially in order:

```bash
# Stage 1: Clean and curate the dataset
python 01_data_cleaning_and_curation.py

# Stage 2: Train baseline model
python 02_baseline_model_training.py

# Stage 3: Optimize hyperparameters (~10-15 minutes)
python 03_hyperparameter_optimization.py

# Stage 4: Train optimized model
python 04_optimized_model_training.py

# Stage 5: External validation
python 05_external_validation.py

# Stage 6: FDA drug repurposing screen
python 06_fda_drug_repurposing.py
```

All random seeds are fixed at 42 for full reproducibility.

## Output Files

| File | Stage | Description |
|------|-------|-------------|
| `clean_dataset_bm_annotated.csv` | 1 | Cleaned dataset with Bemis-Murcko scaffolds |
| `final_dataset_without_confirmatory_data.csv` | 1 | Training pool (193,089 compounds) |
| `confirmatory_holdout_set.csv` | 1 | External validation holdout (1,146 compounds) |
| `phase1_explicit_model.pkl` | 2/4 | Trained XGBoost model |
| `training_features_matrix.pkl` | 2/4 | Training feature matrix (for AD computation) |
| `train_indices.npy`, `test_indices.npy` | 2 | Scaffold-stratified split indices |
| `shap_decoded_interpretable_features.csv` | 2/4 | SHAP feature importance rankings |
| `optuna_best_params.json` | 3 | Optimized hyperparameters |
| `confirmatory_validation_results.csv` | 5 | External validation predictions + metrics |
| `fda_predictions_NOVEL_candidates.csv` | 6 | All novel FDA candidates with flags |
| `priority_cns_candidates_explicit.csv` | 6 | Top 20 CNS-permeable priority candidates |

## Key Methodology

- **Feature representation**: 91-dimensional vector (85 RDKit fragment counts + 6 physicochemical descriptors)
- **Activity label**: Dual-criteria (WT inhibition ≥ 57.94% AND curator outcome = "Active")
- **Data split**: Scaffold-stratified 80/20 (zero scaffold overlap between train/test)
- **Class imbalance**: Handled via `scale_pos_weight` (not SMOTE)
- **Hyperparameter optimization**: Optuna TPE, 50 trials, 3-fold CV, PR-AUC objective
- **Interpretability**: SHAP TreeExplainer on 500 test-set compounds
- **External validation**: Spearman ρ against measured IC₅₀ and selectivity ratio
- **FDA filtering**: 5-stage cascade (AD → novelty → cationic exclusion → probability → CNS permeability)

## Citation

If you use this code or data, please cite:

```
Senary, A.M. & Soliman, A.M. MitoFusionAI: A Machine Learning Framework for Predicting 
Mitochondrial Disruption and Prioritizing FDA-Approved Drug Repurposing Candidates. 
Department of Pharmaceutical Chemistry, The British University in Egypt.
```

## License

This project is intended for academic research purposes.

## Contact

- A.M. Senary — [ahmedsenary235@gmail.com](mailto:ahmedsenary235@gmail.com)
- Department of Pharmaceutical Chemistry, Faculty of Pharmacy, The British University in Egypt, El-Sherouk City, Cairo 11837, Egypt
