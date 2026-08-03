"""
MitoFusionAI Pipeline — 04 Optimized Model Training
Source notebook: training_model_with_optimization.ipynb
"""

# ==================== REPRODUCIBILITY SETUP ====================
import random
import numpy as np
import os

# Set all random seeds for reproducibility
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

print(f"Random seed set to {SEED} for reproducibility")
print("=" * 60)

# ==================== IMPORTS ====================
import warnings
warnings.filterwarnings('ignore')
from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')

import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
# [NEW ENHANCEMENT] Import built-in named fragment library
from rdkit.Chem import Fragments
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, matthews_corrcoef, confusion_matrix, roc_curve
import xgboost as xgb
import joblib
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

print("=== 1. LOAD CLEANED DATA ===")
df = pd.read_csv('final_dataset_without_confirmatory_data.csv')

print("=== 2. SCAFFOLD SPLIT ===")
def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold) if scaffold else None
    return None

df['scaffold'] = df['smiles'].apply(get_scaffold)
df = df.dropna(subset=['scaffold'])

scaffold_to_indices = defaultdict(list)
for idx, scaffold in enumerate(df['scaffold']):
    scaffold_to_indices[scaffold].append(idx)

scaffold_groups = list(scaffold_to_indices.values())
np.random.seed(SEED)
np.random.shuffle(scaffold_groups)

train_idx, test_idx = [], []
for group in scaffold_groups:
    if len(test_idx) < len(df) * 0.2:
        test_idx.extend(group)
    else:
        train_idx.extend(group)

print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

print("=== 3. CONVERT TO FEATURES (EXPLICIT MOIETY ARCHITECTURE) ===")
# Extract all 85 built-in named fragment functions
fragment_functions = [name for name in dir(Fragments) if name.startswith('fr_')]

# Create clean, human-readable feature names for the XGBoost model
FEATURE_NAMES = [name.replace('fr_', 'Moiety_') for name in fragment_functions] + \
                ['MolWt', 'MolLogP', 'TPSA', 'NumRotatableBonds', 'NumHAcceptors', 'NumHDonors']

def smiles_to_features(smiles):
    """
    Instead of abstract Morgan bits, we physically count the presence 
    of 85 specific, named chemical moieties and combine them with physical properties.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # 1. Explicit Named Chemical Moieties (e.g., benzene, halogens, amides)
        moieties = [getattr(Fragments, func_name)(mol) for func_name in fragment_functions]
        
        # 2. Dense Physiochemical Descriptors
        mw = Descriptors.MolWt(mol)                    
        logp = Descriptors.MolLogP(mol)                
        tpsa = Descriptors.TPSA(mol)                   
        rot_bonds = Descriptors.NumRotatableBonds(mol) 
        hba = Descriptors.NumHAcceptors(mol)           
        hbd = Descriptors.NumHDonors(mol)              
        
        # Combine them into a single human-readable vector (Length: 91)
        return moieties + [mw, logp, tpsa, rot_bonds, hba, hbd]
    return None

# Generate features and map them directly to a DataFrame using our column names
X = pd.DataFrame([smiles_to_features(s) for s in df['smiles']], columns=FEATURE_NAMES)
y = df['active'].values

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print("=== 4. TRAIN XGBOOST ===")
raw_imbalance_ratio = len(y_train[y_train==0]) / len(y_train[y_train==1])
print(f"Raw Class Imbalance Ratio: {raw_imbalance_ratio:.2f}:1")

# [OPTIMIZED] XGBoost Initialization from Optuna Bayesian Search

import json
with open('optuna_best_params.json') as f:
    best_params = json.load(f)
best_params.update({'random_state': SEED, 'tree_method': 'hist', 'verbosity': 0})
model = xgb.XGBClassifier(**best_params)
model.fit(X_train, y_train)

print("=== 5. EVALUATE ===")
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"Scaffold-split AUC: {auc:.3f}")

# ============================================================
# COMPREHENSIVE METRICS & HIGH-PRECISION THRESHOLDING
# ============================================================
print("\n" + "="*70)
print("COMPREHENSIVE MODEL VALIDATION")
print("="*70)

pr_auc = average_precision_score(y_test, y_pred_proba)

precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)

target_precision = 0.50 
valid_indices = np.where(precision_vals[:-1] >= target_precision)[0]

if len(valid_indices) > 0:
    strict_threshold = thresholds_pr[valid_indices[0]] 
else:
    strict_threshold = np.percentile(y_pred_proba, 99) 

y_pred_strict = (y_pred_proba >= strict_threshold).astype(int)
mcc = matthews_corrcoef(y_test, y_pred_strict)

cm = confusion_matrix(y_test, y_pred_strict)
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0 

print(f"\n📊 CORE METRICS:")
print(f"   ROC-AUC:               {auc:.4f}")
print(f"   PR-AUC (Avg Precision): {pr_auc:.4f}")
print(f"   MCC:                   {mcc:.4f}")
print(f"   Strict Threshold:      {strict_threshold:.4f} (Optimized for Virtual Screening)")

print(f"\n📈 CLASSIFICATION REPORT (Threshold = {strict_threshold:.4f}):")
print(f"   Precision:             {precision:.4f} (When model says hit, it is correct {precision*100:.1f}% of the time)")
print(f"   Recall/Sensitivity:    {recall:.4f}")

# ============================================================
# UPGRADED SHAP ANALYSIS (FULLY INTERPRETABLE MOIETY OUTPUT)
# ============================================================
print("\n" + "="*70)
print("STANDARD SHAP ANALYSIS - EXPLICIT MOIETY DECODING")
print("="*70)

try:
    import shap
    
    # 1. Create explainer
    sample_size = min(500, len(X_test))
    X_sample = X_test.iloc[:sample_size].astype(float)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.array(shap_values)
    
    # 2. Global feature importance
    shap_importance = pd.DataFrame({
        'feature_name': FEATURE_NAMES,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    # 3. Decode features (No Morgan bits needed, just mapping names directly)
    print("\nExtracting Top Interpretable Features...")
    
    top_features_decoded = []
    
    for i, row in shap_importance.head(25).iterrows():
        feat_name = row['feature_name']
        importance = row['importance']
        
        # Determine the type of feature based on the naming convention we set up
        if feat_name.startswith('Moiety_'):
            feat_type = 'Structural_Moiety'
        else:
            feat_type = 'Physiochemical_Property'
            
        top_features_decoded.append({
            'feature_type': feat_type,
            'descriptor_name': feat_name,
            'shap_importance': importance
        })
                
    if top_features_decoded:
        decoded_df = pd.DataFrame(top_features_decoded)
        decoded_df.to_csv('shap_decoded_interpretable_features.csv', index=False)
        print("✓ Decoded chemical and physical features saved")
        
        print("\n🎯 TOP 10 DRIVERS OF DRP1 INHIBITION:")
        for idx, row in decoded_df.head(10).iterrows():
            print(f"  {idx+1}. {row['descriptor_name']:20s} | Type: {row['feature_type']:25s} | Importance: {row['shap_importance']:.4f}")

    # Visualizations
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=15)
    axes1[0].set_title("A) Top Explicit Feature Importance", fontweight='bold')
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    axes1[1].set_title("B) Feature Impact Distribution", fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_explicit_analysis.png', dpi=300, bbox_inches='tight')

except Exception as e:
    print(f"⚠️ SHAP analysis failed: {str(e)}")

# ============================================================
# SAVE TRAINING DATA FOR FDA PREDICTION (AD PREPARATION)
# ============================================================
print("\n=== 6. SAVE MODEL AND TRAINING DATA ===")
joblib.dump(model, 'phase1_explicit_model.pkl')

np.save('train_indices.npy', train_idx)
np.save('test_indices.npy', test_idx)

# Save the structured NumPy array (now 91 columns instead of 1030, making it much faster!)
train_features_array = X_train.values.astype(np.float32)
with open('training_features_matrix.pkl', 'wb') as f:
    pickle.dump(train_features_array, f)

with open('training_smiles.pkl', 'wb') as f:
    pickle.dump(df['smiles'].iloc[train_idx].tolist(), f)

print(f"✓ Model saved as 'phase1_explicit_model.pkl'")
print(f"✓ Training Matrix saved for FDA vectorization ({train_features_array.shape[0]} compounds, {train_features_array.shape[1]} features)")

print("\n" + "="*70)
print("PIPELINE COMPLETE - OPTIMIZED FOR FDA REPURPOSING")
print("="*70)
