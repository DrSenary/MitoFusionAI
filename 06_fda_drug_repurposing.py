"""
MitoFusionAI Pipeline — 06 Fda Drug Repurposing
Source notebook: FDA_Approved_repurposing_script_FIXED.ipynb
"""

# ============================================================
# MITOFUSIONAI V3 - FDA DRUG PREDICTION (FILTERED & FIXED)
# ============================================================

# ==================== 0. CONFIGURATION ====================
# --- Input files ---
SDF_FILE                = "fda_drugs.sdf"
MODEL_FILE              = "phase1_explicit_model.pkl"
TRAINING_MATRIX_FILE    = "training_features_matrix.pkl"
TRAINING_CSV            = "final_dataset_without_confirmatory_data.csv"  # for CID de-duplication
TRAINING_CID_COLUMN     = "pubchem_id"  # column in TRAINING_CSV holding PubChem CIDs

# --- Filter thresholds ---
# An FDA drug is flagged 'in_training_set' if EITHER:
#   (a) its PubChem CID appears in the training CSV, OR
#   (b) its feature-space Euclidean distance to the nearest training compound < NOVELTY_DISTANCE_THRESHOLD
NOVELTY_DISTANCE_THRESHOLD = 0.05    # features are 91-dim; < 0.05 ≈ near-identical

# Reject permanently-cationic species (PAINS-like in yeast growth assays)
EXCLUDE_CATIONIC_AMPHIPHILES = True

# Flag (don't reject) phenothiazine-class compounds — they are a *known* hit class
# from the training data and should be reported separately, not as "novel" findings.
FLAG_PHENOTHIAZINE       = True
PHENOTHIAZINE_SMARTS     = "c1ccc2c(c1)Nc1ccccc1S2"   # the CORRECT pattern from the conversation

# Optionally fetch real drug names from PubChem REST API (requires internet)
PUBCHEM_NAME_LOOKUP      = False

# ==================== SUPPRESS WARNINGS ====================
import os, logging, warnings, time, json, urllib.request
warnings.filterwarnings('ignore')
logging.getLogger('rdkit').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem import Descriptors, Lipinski, Fragments
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import PandasTools
from rdkit import Chem
from sklearn.neighbors import NearestNeighbors
import pickle
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd

print("""
================================================================
       MITOFUSIONAI V3 - FDA DRUG PREDICTION (FIXED)
       With CID de-dup, cationic exclusion, novelty filter
================================================================
""")

# ==================== 1. EXPLICIT MOIETY FEATURES ====================
print("[1/11] Setting up explicit chemistry features...")
fragment_functions = [name for name in dir(Fragments) if name.startswith('fr_')]
FEATURE_NAMES = [name.replace('fr_', 'Moiety_') for name in fragment_functions] + \
                ['MolWt', 'MolLogP', 'TPSA', 'NumRotatableBonds', 'NumHAcceptors', 'NumHDonors']

def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol:
        moieties = [getattr(Fragments, fn)(mol) for fn in fragment_functions]
        mw       = Descriptors.MolWt(mol)
        logp     = Descriptors.MolLogP(mol)
        tpsa     = Descriptors.TPSA(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        hba      = Descriptors.NumHAcceptors(mol)
        hbd      = Descriptors.NumHDonors(mol)
        return moieties + [mw, logp, tpsa, rot_bonds, hba, hbd]
    return None

# ==================== 2. CHEMISTRY-AWARE FLAG FUNCTIONS ====================
print("[2/11] Defining chemistry-aware flag functions...")

_PHENO_PATTERN = Chem.MolFromSmarts(PHENOTHIAZINE_SMARTS) if FLAG_PHENOTHIAZINE else None
# Sanity check: the SMARTS must match a real phenothiazine, otherwise the fix is broken.
if _PHENO_PATTERN is not None:
    _test_mol = Chem.MolFromSmiles('CN1c2ccccc2Sc2ccccc21')   # promazine
    assert _test_mol is not None and _test_mol.HasSubstructMatch(_PHENO_PATTERN), \
        "Phenothiazine SMARTS self-test FAILED — fix the pattern before continuing."
    print("   Phenothiazine SMARTS self-test passed (matches promazine).")

def is_permanently_cationic(neutral_mol):
    """Return True if the molecule carries a permanent positive charge that survives
    Uncharger neutralization (quaternary ammonium, pyridinium, sulfonium, etc.).
    These are well-documented PAINS/aggregator false-positives in yeast growth assays
    because they non-specifically disrupt membranes (including mitochondrial membranes)."""
    if neutral_mol is None:
        return False
    for atom in neutral_mol.GetAtoms():
        if atom.GetFormalCharge() > 0:
            return True
    return False

def is_phenothiazine(neutral_mol):
    if _PHENO_PATTERN is None or neutral_mol is None:
        return False
    return neutral_mol.HasSubstructMatch(_PHENO_PATTERN)

def lookup_pubchem_name(cid, timeout=5):
    """Optional helper: fetch a drug name from PubChem REST API. Returns None on any failure."""
    if not cid or cid <= 0:
        return None
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{int(cid)}/property/IUPACName/TXT"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            name = resp.read().decode('utf-8').strip()
            return name if name and 'error' not in name.lower() else None
    except Exception:
        return None

# ==================== 3. LOAD TRAINED MODEL ====================
print("\n[3/11] Loading trained model...")
model = joblib.load(MODEL_FILE)
print(f"   Model loaded: {MODEL_FILE}")

# ==================== 4. LOAD TRAINING MATRIX + TRAINING CSV (for de-dup) ====================
print("\n[4/11] Loading training matrix and training CSV...")
with open(TRAINING_MATRIX_FILE, 'rb') as f:
    train_features_array = pickle.load(f)
print(f"   Training matrix: {train_features_array.shape[0]} compounds x {train_features_array.shape[1]} features")

print("   Fitting Nearest Neighbors space for Applicability Domain...")
ad_model = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1)
ad_model.fit(train_features_array)

# Load training CSV to get the set of PubChem CIDs we trained on
train_cids = set()
try:
    train_df_for_dedup = pd.read_csv(TRAINING_CSV)
    if TRAINING_CID_COLUMN not in train_df_for_dedup.columns:
        # Try common alternatives
        for alt in ['PUBCHEM_CID', 'CID', 'cid', 'ID']:
            if alt in train_df_for_dedup.columns:
                train_df_for_dedup = train_df_for_dedup.rename(columns={alt: TRAINING_CID_COLUMN})
                print(f"   (renamed '{alt}' -> '{TRAINING_CID_COLUMN}' in training CSV)")
                break
    if TRAINING_CID_COLUMN in train_df_for_dedup.columns:
        train_cids = set(
            pd.to_numeric(train_df_for_dedup[TRAINING_CID_COLUMN], errors='coerce')
              .dropna().astype(int).unique()
        )
        print(f"   Loaded {len(train_cids):,} unique PubChem CIDs from training pool for de-duplication")
    else:
        print(f"   WARNING: no '{TRAINING_CID_COLUMN}' column in {TRAINING_CSV}; CID de-dup will be skipped.")
        print(f"   Available columns: {list(train_df_for_dedup.columns)[:10]}")
except FileNotFoundError:
    print(f"   WARNING: {TRAINING_CSV} not found — CID de-dup will be SKIPPED.")
    print(f"   Set TRAINING_CSV correctly or move the file next to this notebook.")
except Exception as e:
    print(f"   WARNING: could not load training CSV ({e}); CID de-dup will be SKIPPED.")

# ==================== 5. LOAD SDF ====================
print(f"\n[5/11] Loading FDA drugs from SDF: {SDF_FILE}")
sdf_df = PandasTools.LoadSDF(SDF_FILE, smilesName='SMILES', molColName='Molecule')
print(f"   Loaded {len(sdf_df):,} compounds from '{SDF_FILE}'")

if 'SMILES' not in sdf_df.columns:
    for col in sdf_df.columns:
        if 'smiles' in col.lower():
            sdf_df = sdf_df.rename(columns={col: 'SMILES'})
            break

# --- Identify the PubChem CID column (was silently used as 'drug_name' in original code) ---
cid_col = None
for cand in ['PUBCHEM_CID', 'PubChem_CID', 'CID', 'cid', 'pubchem_id', 'ID']:
    if cand in sdf_df.columns:
        cid_col = cand
        break
if cid_col is None:
    sdf_df['pubchem_cid'] = [0] * len(sdf_df)
    print("   WARNING: no CID column found in SDF; CID de-duplication will not work.")
else:
    sdf_df['pubchem_cid'] = pd.to_numeric(sdf_df[cid_col], errors='coerce').fillna(0).astype(int)
    print(f"   Using '{cid_col}' as PubChem CID source.")

# --- Try to grab a real drug name from the SDF (if any name-like field exists) ---
name_col = None
for cand in ['Name', 'DRUG_NAME', 'Generic_Name', 'GENERIC_NAME', 'TRADE_NAME', 'DRUGBANK_NAME']:
    if cand in sdf_df.columns:
        name_col = cand
        print(f"   Using '{name_col}' as drug-name source.")
        break
if name_col is None:
    print("   No real drug-name field in SDF — 'drug_name' will fall back to 'CID_<number>' strings.")
    print("   (Optionally set PUBCHEM_NAME_LOOKUP = True to fetch IUPAC names from PubChem.)")

# ==================== 6. EXTRACT FEATURES + CHEMISTRY FLAGS ====================
print("\n[6/11] Neutralizing and computing features + chemistry flags for FDA drugs...")
uncharger = rdMolStandardize.Uncharger()

fda_features, fda_cids, fda_names, fda_smiles = [], [], [], []
fda_is_cationic, fda_is_phenothiazine, fda_in_training_by_cid = [], [], []

for idx, row in tqdm(sdf_df.iterrows(), total=len(sdf_df)):
    smiles = row['SMILES']
    cid = int(row['pubchem_cid']) if pd.notna(row['pubchem_cid']) else 0
    if name_col and pd.notna(row.get(name_col, None)):
        name = str(row[name_col])
    elif cid > 0:
        name = f"CID_{cid}"
    else:
        name = f"Drug_{idx}"

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        continue
    try:
        neutral_mol = uncharger.uncharge(mol)
        neutral_smiles = Chem.MolToSmiles(neutral_mol)
        feats = smiles_to_features(neutral_smiles)
        if feats is None:
            continue
        fda_features.append(feats)
        fda_cids.append(cid)
        fda_names.append(name)
        fda_smiles.append(neutral_smiles)
        fda_is_cationic.append(is_permanently_cationic(neutral_mol))
        fda_is_phenothiazine.append(is_phenothiazine(neutral_mol))
        fda_in_training_by_cid.append(cid in train_cids if cid > 0 else False)
    except Exception:
        continue

X_fda = np.array(fda_features)
print(f"   Extracted {X_fda.shape[1]} features for {len(fda_features):,} valid FDA drugs")

# ==================== 7. APPLICABILITY DOMAIN + NOVELTY FILTER ====================
print("\n[7/11] Calculating Applicability Domain and novelty filter...")
distances, _ = ad_model.kneighbors(X_fda)
distances = distances.flatten()

distance_threshold = np.percentile(distances, 95)
in_ad = (distances <= distance_threshold).astype(int)

# Novelty filter: also flag distance-based "in training" (catches structural look-alikes)
in_training_by_distance = (distances < NOVELTY_DISTANCE_THRESHOLD).astype(int)
in_training_set = np.maximum(
    np.array(fda_in_training_by_cid, dtype=int),
    in_training_by_distance
)

n_cid_match   = int(sum(fda_in_training_by_cid))
n_dist_match  = int(in_training_by_distance.sum())
n_in_training = int(in_training_set.sum())
print(f"   {n_cid_match} FDA drugs matched by PubChem CID against training pool")
print(f"   {n_dist_match} FDA drugs matched by structural distance (< {NOVELTY_DISTANCE_THRESHOLD})")
print(f"   {n_in_training} FDA drugs flagged as in_training_set (union of both)")
print(f"   {in_ad.sum()} FDA drugs inside the structural Applicability Domain")

# ==================== 8. PREDICTIONS ====================
print("\n[8/11] Making predictions...")
predictions = model.predict_proba(X_fda)[:, 1]
confidence_scores = np.abs(predictions - 0.5) * 2

# ==================== 9. ASSEMBLE RESULTS WITH ALL FLAGS ====================
print("\n[9/11] Assembling results with all chemistry flags...")
results = pd.DataFrame(X_fda, columns=FEATURE_NAMES)
results.insert(0, 'pubchem_cid', fda_cids)
results.insert(1, 'drug_name', fda_names)
results.insert(2, 'smiles', fda_smiles)
results.insert(3, 'predicted_probability', predictions)
results.insert(4, 'confidence_score', confidence_scores)
results.insert(5, 'distance_to_training', distances)
results.insert(6, 'in_applicability_domain', in_ad)
results.insert(7, 'in_training_set', in_training_set)
results.insert(8, 'is_cationic_amphiphile', np.array(fda_is_cationic, dtype=int))
results.insert(9, 'is_phenothiazine', np.array(fda_is_phenothiazine, dtype=int))

# CNS permeability (Lipinski-like, conservative)
results['CNS_permeable'] = (
    (results['MolWt'] <= 450) &
    (results['MolLogP'] <= 5) &
    (results['NumHDonors'] <= 3) &
    (results['TPSA'] <= 90)
).astype(int)

# Master "novel candidate" flag — the one you actually want to report in the paper
is_novel = (
    (results['in_training_set'] == 0) &
    (results['is_cationic_amphiphile'] == 0 if EXCLUDE_CATIONIC_AMPHIPHILES else True)
)
results.insert(10, 'is_novel_candidate', is_novel.astype(int))

# Optional PubChem name lookup (slow, but only needed once)
if PUBCHEM_NAME_LOOKUP:
    print("   Fetching IUPAC names from PubChem REST API (this can take a few minutes)...")
    for i, cid in enumerate(tqdm(fda_cids)):
        if cid > 0 and fda_names[i].startswith('CID_'):
            n = lookup_pubchem_name(cid)
            if n:
                fda_names[i] = n
    results['drug_name'] = fda_names

# Sort by probability (highest first); add rank over the FULL set
results = results.sort_values('predicted_probability', ascending=False).reset_index(drop=True)
results.insert(0, 'rank', np.arange(1, len(results) + 1))

# Also add a rank among the novel candidates only
novel_df = results[results['is_novel_candidate'] == 1].copy()
novel_df['rank_novel'] = np.arange(1, len(novel_df) + 1)
results = results.merge(
    novel_df[['rank_novel']], left_index=True, right_index=True, how='left'
)
results['rank_novel'] = results['rank_novel'].astype('Int64')

# ==================== 10. SAVE OUTPUTS ====================
print("\n[10/11] Saving CSV outputs...")
results.to_csv('fda_predictions_explicit_complete.csv', index=False)
results.head(100).to_csv('fda_predictions_explicit_top100.csv', index=False)

# THE filtered candidate set — what you should actually report in the paper
novel_df.to_csv('fda_predictions_NOVEL_candidates.csv', index=False)

# Priority CNS candidates — only from the novel pool
priority_cns = novel_df[
    (novel_df['CNS_permeable'] == 1) &
    (novel_df['predicted_probability'] > 0.5) &
    (novel_df['in_applicability_domain'] == 1)
].head(20)
priority_cns.to_csv('priority_cns_candidates_explicit.csv', index=False)

print(f"   fda_predictions_explicit_complete.csv     — all {len(results):,} predictions (with flags)")
print(f"   fda_predictions_explicit_top100.csv       — top 100 by probability (with flags)")
print(f"   fda_predictions_NOVEL_candidates.csv      — {len(novel_df):,} novel candidates (filtered)")
print(f"   priority_cns_candidates_explicit.csv      — top {len(priority_cns)} CNS-permeable novel candidates")

# Print a quick summary so the user immediately sees what the filters did
print("\n" + "=" * 80)
print("FILTER IMPACT SUMMARY")
print("=" * 80)
print(f"  Total FDA drugs in SDF                    : {len(sdf_df):>6,}")
print(f"  Valid compounds (parsed + neutralized)    : {len(fda_features):>6,}")
print(f"  In training set (by CID or distance)      : {n_in_training:>6,}")
print(f"  Permanently cationic amphiphiles          : {int(sum(fda_is_cationic)):>6,}")
print(f"  Phenothiazine-class compounds             : {int(sum(fda_is_phenothiazine)):>6,}")
print(f"  NOVEL candidates (passes all filters)     : {len(novel_df):>6,}")
print(f"  Novel + CNS-permeable + prob > 0.5 + inAD : {len(priority_cns):>6,}")
print()
print("Top 10 NOVEL candidates (the list to actually look at for repurposing):")
print(novel_df[['rank_novel', 'pubchem_cid', 'drug_name', 'predicted_probability',
                'distance_to_training', 'MolLogP', 'MolWt', 'CNS_permeable',
                'is_phenothiazine']].head(10).to_string(index=False))

# ==================== 11. VISUALIZATIONS ====================
print("\n[11/11] Generating visualizations...")
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(20, 16))

# Plot 1: Prediction Distribution, split by is_novel
ax1 = plt.subplot(2, 2, 1)
ax1.hist(results.loc[results['is_novel_candidate'] == 1, 'predicted_probability'],
         bins=40, color='green', alpha=0.6, label='Novel candidates', edgecolor='black')
ax1.hist(results.loc[results['is_novel_candidate'] == 0, 'predicted_probability'],
         bins=40, color='red', alpha=0.5, label='Filtered out (training/cationic)', edgecolor='black')
ax1.axvline(x=0.5, color='black', linestyle='--', linewidth=1)
ax1.set_title('Prediction Distribution: Novel vs Filtered-out', fontweight='bold')
ax1.set_xlabel('Predicted Probability')
ax1.set_ylabel('Number of FDA Drugs')
ax1.legend()

# Plot 2: Applicability Domain — color by is_novel
ax2 = plt.subplot(2, 2, 2)
novel_mask = results['is_novel_candidate'] == 1
ax2.scatter(results.loc[~novel_mask, 'distance_to_training'],
            results.loc[~novel_mask, 'predicted_probability'],
            c='red', alpha=0.4, s=20, label='Filtered out')
sc = ax2.scatter(results.loc[novel_mask, 'distance_to_training'],
                 results.loc[novel_mask, 'predicted_probability'],
                 c=results.loc[novel_mask, 'MolLogP'], cmap='viridis',
                 alpha=0.8, s=25, label='Novel candidates')
ax2.axvline(x=distance_threshold, color='blue', linestyle='--', label='AD boundary (95th pct)')
ax2.axvline(x=NOVELTY_DISTANCE_THRESHOLD, color='orange', linestyle=':',
            label=f'Novelty threshold ({NOVELTY_DISTANCE_THRESHOLD})')
ax2.set_title('Applicability Domain vs Probability', fontweight='bold')
ax2.set_xlabel('Euclidean Distance to Nearest Training Compound')
ax2.set_ylabel('Predicted Probability')
ax2.set_xscale('symlog', linthresh=0.01)
plt.colorbar(sc, ax=ax2, label='MolLogP (novel only)')
ax2.legend(loc='lower right', fontsize=8)

# Plot 3: Top 20 NOVEL candidates (not the unfiltered top 20)
ax3 = plt.subplot(2, 2, 3)
top_novel = novel_df.head(20).copy()
y_pos = np.arange(len(top_novel))
colors = ['green' if cns else 'steelblue' for cns in top_novel['CNS_permeable']]
ax3.barh(y_pos, top_novel['predicted_probability'], color=colors, edgecolor='black')
labels = [f"{n[:20]} (CID {c})" for n, c in zip(top_novel['drug_name'], top_novel['pubchem_cid'])]
ax3.set_yticks(y_pos)
ax3.set_yticklabels(labels)
ax3.invert_yaxis()
ax3.set_title('Top 20 NOVEL Candidates (Green = CNS-permeable)', fontweight='bold')
ax3.set_xlabel('Predicted Probability')
ax3.axvline(x=0.5, color='red', linestyle='--', linewidth=1)

# Plot 4: Filtered-out breakdown — counts by reason
ax4 = plt.subplot(2, 2, 4)
reasons = {
    'In training set\n(CID match)':   int(results['in_training_set'].sum()),
    'Cationic\namphiphile':           int(results['is_cationic_amphiphile'].sum()),
    'Phenothiazine\n(known class)':   int(results['is_phenothiazine'].sum()),
    'Outside AD':                     int((results['in_applicability_domain'] == 0).sum()),
    'Novel candidates\n(passed all)': int(results['is_novel_candidate'].sum()),
}
bars = ax4.bar(range(len(reasons)), list(reasons.values()),
               color=['red', 'orange', 'gold', 'grey', 'green'], edgecolor='black')
ax4.set_xticks(range(len(reasons)))
ax4.set_xticklabels(list(reasons.keys()), fontsize=9)
ax4.set_title('Filter Breakdown', fontweight='bold')
ax4.set_ylabel('Number of FDA Drugs')
for bar, v in zip(bars, reasons.values()):
    ax4.text(bar.get_x() + bar.get_width()/2, v + max(reasons.values())*0.01,
             f'{v}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('fda_analysis_explicit.png', dpi=300)
print("   Visualizations saved to 'fda_analysis_explicit.png'")

print("\n" + "=" * 80)
print("V3 PIPELINE (FIXED) COMPLETE")
print("=" * 80)
print("Next steps:")
print("  1. Open fda_predictions_NOVEL_candidates.csv — this is your actual hit list")
print("  2. Manually verify top-10 novel CIDs against PubChem (drug names, indications)")
print("  3. If drug_name column is still 'CID_<number>' strings, run with PUBCHEM_NAME_LOOKUP = True")
print("     OR fill names manually — they're required for the paper")
print("  4. Cross-check the novel hits against known mito-fusion inhibitor literature")
print("  5. If you want stricter curation, replace fda_drugs.sdf with the DrugBank")
print("     FDA-approved subset (cleans out surfactants/dyes/antiseptics)")

