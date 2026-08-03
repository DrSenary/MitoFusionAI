# ==================== EXTERNAL VALIDATION AGAINST AID 1361 ====================
# Run this AFTER the final, Optuna-optimized model is trained (Step 4), and
# BEFORE the FDA repurposing screen (Step 6). This is the model's one honest
# check against real, wet-lab-confirmed dose-response data rather than more
# of the same single-concentration primary-screen labels it was trained on.

import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

# ---------------------------------------------------------------
# 1. Load the finalized model and the untouched confirmatory holdout
# ---------------------------------------------------------------
print("[1/5] Loading final model and confirmatory holdout set...")
model = joblib.load('phase1_explicit_model.pkl')

holdout = pd.read_csv('confirmatory_holdout_set.csv', low_memory=False)

confirmatory = pd.read_csv('AID_1361_datatable.csv', skiprows=[1, 2, 3, 4], low_memory=False)
confirmatory = confirmatory.rename(columns={'PUBCHEM_SID': 'pubchem_id'})

# Merge holdout compounds with their real dose-response ground truth
merged = holdout.merge(
    confirmatory[['pubchem_id', 'PUBCHEM_ACTIVITY_OUTCOME', 'WT IC50', 'Mutant IC50']],
    on='pubchem_id', how='inner'
)
print(f"✓ Matched {len(merged)} of {len(holdout)} holdout compounds to confirmatory dose-response data.")

# ---------------------------------------------------------------
# 2. Featurize the holdout compounds (identical feature pipeline as training)
# ---------------------------------------------------------------
print("\n[2/5] Generating features for holdout compounds...")
fragment_functions = [name for name in dir(Fragments) if name.startswith('fr_')]
FEATURE_NAMES = [name.replace('fr_', 'Moiety_') for name in fragment_functions] + \
                ['MolWt', 'MolLogP', 'TPSA', 'NumRotatableBonds', 'NumHAcceptors', 'NumHDonors']


def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    moieties = [getattr(Fragments, fn)(mol) for fn in fragment_functions]
    return moieties + [
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol), Descriptors.NumHAcceptors(mol), Descriptors.NumHDonors(mol),
    ]


feats = [smiles_to_features(s) for s in merged['smiles']]
valid_mask = [f is not None for f in feats]
merged = merged[valid_mask].reset_index(drop=True)
X_holdout = pd.DataFrame([f for f in feats if f is not None], columns=FEATURE_NAMES)
print(f"✓ Featurized {len(X_holdout)} compounds.")

# ---------------------------------------------------------------
# 3. Predict
# ---------------------------------------------------------------
print("\n[3/5] Running model predictions on holdout...")
merged['predicted_probability'] = model.predict_proba(X_holdout)[:, 1]

# ---------------------------------------------------------------
# 4. Classification metrics against the REAL confirmatory outcome
# ---------------------------------------------------------------
print("\n[4/5] Classification performance vs. confirmatory dose-response outcome:")
y_true = (merged['PUBCHEM_ACTIVITY_OUTCOME'] == 'Active').astype(int)
y_score = merged['predicted_probability']

auc = roc_auc_score(y_true, y_score)
pr_auc = average_precision_score(y_true, y_score)
y_pred_binary = (y_score >= 0.5).astype(int)
prec = precision_score(y_true, y_pred_binary, zero_division=0)
rec = recall_score(y_true, y_pred_binary, zero_division=0)

print(f"   ROC-AUC:   {auc:.4f}")
print(f"   PR-AUC:    {pr_auc:.4f}")
print(f"   Precision: {prec:.4f}  (threshold = 0.5)")
print(f"   Recall:    {rec:.4f}  (threshold = 0.5)")

# ---------------------------------------------------------------
# 5. Does model confidence track real potency/selectivity?
# ---------------------------------------------------------------
print("\n[5/5] Rank correlation between model confidence and real potency/selectivity:")

potency_data = merged.dropna(subset=['WT IC50'])
if len(potency_data) > 1:
    rho_potency, p_potency = spearmanr(potency_data['predicted_probability'], potency_data['WT IC50'])
    print(f"   Spearman(predicted_probability, WT IC50): rho = {rho_potency:.3f}, p = {p_potency:.4f}")
    print("   (Negative rho is the expected direction: higher confidence <-> lower IC50 <-> more potent.)")
else:
    print("   Not enough compounds with a determinable WT IC50 to compute correlation.")

selectivity_data = merged.dropna(subset=['WT IC50', 'Mutant IC50']).copy()
if len(selectivity_data) > 1:
    selectivity_data['selectivity_ratio'] = selectivity_data['Mutant IC50'] / selectivity_data['WT IC50']
    rho_sel, p_sel = spearmanr(selectivity_data['predicted_probability'], selectivity_data['selectivity_ratio'])
    print(f"   Spearman(predicted_probability, selectivity ratio): rho = {rho_sel:.3f}, p = {p_sel:.4f}")
    print("   (Positive rho is the expected direction: higher confidence <-> more selective for WT over mutant.)")
else:
    print("   Not enough compounds with both IC50s determinable to compute selectivity correlation.")

merged.to_csv('confirmatory_validation_results.csv', index=False)
print("\n✓ Full results saved to 'confirmatory_validation_results.csv'.")
