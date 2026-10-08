# MitoFusionAI — Reproducibility Sample Data

This directory contains a stratified sample of the MitoFusionAI training dataset, released to support reproducibility of the manuscript:

> Senary AM, Soliman AM. *Machine learning prediction of mitochondrial disruptors from yeast phenotypic screening for FDA approved drug repurposing using MitoFusionAI.* Discover Chemistry (2026).

## Files

| File | Description | Size |
|------|-------------|------|
| `sample_training_set_1000.csv` | 1,000 compounds drawn from PubChem BioAssay AID 1362 (the primary yeast mitochondrial-fusion screen), stratified to preserve the 2.07% active-class prevalence of the full 194,235-compound dataset. | 1,000 rows × 6 cols |
| `demo_notebook.ipynb` | End-to-end Jupyter notebook that loads the sample, featurizes 91 descriptors (85 RDKit fragments + 6 physicochemical), performs a scaffold-stratified 80/20 split, trains the XGBoost classifier with the manuscript's optimized hyperparameters, and reports ROC-AUC, PR-AUC, MCC, balanced accuracy. Also generates a SHAP feature-importance plot. | 7 cells |

## How to reproduce

```bash
pip install rdkit xgboost scikit-learn pandas numpy shap matplotlib
jupyter notebook demo_notebook.ipynb
```

Expected metrics on the sample's 20% test split (your numbers will vary slightly due to scaffold-stratification stochasticity):

| Metric | Sample (expected) | Full dataset (manuscript) |
|--------|-------------------|---------------------------|
| ROC-AUC | 0.75–0.85 | 0.849 |
| PR-AUC | 0.05–0.15 | 0.157 |
| MCC | 0.10–0.25 | 0.191 |
| Balanced accuracy | 0.65–0.75 | 0.69 |

The smaller sample produces wider metric variance because the active class (~20 compounds) is too small to give a stable PR-AUC. The full 194,235-compound dataset is hosted on PubChem (AID 1362) and can be downloaded directly:

```
https://pubchem.ncbi.nlm.nih.gov/bioassay/1362
```

The external validation set (AID 1361) is similarly available:

```
https://pubchem.ncbi.nlm.nih.gov/bioassay/1361
```

## Columns in `sample_training_set_1000.csv`

| Column | Type | Description |
|--------|------|-------------|
| `pubchem_id` | int | PubChem Compound ID |
| `canonical_smiles` | str | RDKit-canonicalized SMILES string |
| `activity_outcome` | str | Curator-assigned outcome (Active/Inactive) |
| `WT_Inhibition_Avg_10uM` | float | Wild-type strain growth inhibition (%) at 10 µM |
| `active_label` | int | Dual-criteria label (1 = active: WT ≥ 57.94% AND curator Active; 0 = inactive) |
| `scaffold` | str | Bemis–Murcko scaffold SMILES |

## License

Code: MIT. Data: derived from PubChem BioAssay AID 1362 (NIH Molecular Libraries Program, Nunnari laboratory).
