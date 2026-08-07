"""
MitoFusionAI Pipeline — 01 Data Cleaning And Curation
"""

# =====================================================================
# Stage 1: Data Cleaning, Activity Labeling, and Structural Annotation
# =====================================================================
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# ------------------------------------------------------------------
# 1. Pharmacophore dictionary
# ------------------------------------------------------------------

PHARMACOPHORE_SMARTS = {
    # Heterocycles
    'Piperazine_Ring': 'C1CNCCN1',
    'Pyridine_Ring': 'c1ccncc1',
    'Pyrimidine_Ring': 'c1cncnc1',
    'Imidazole_Ring': 'c1cncn1',

    # Functional Groups
    'Amide_Group': 'C(=O)N',
    'Carboxylic_Acid': 'C(=O)O',
    'Ester_Group': 'C(=O)O[C;!H0]',
    'Sulfonamide': 'S(=O)(=O)N',

    # Halogen Patterns (position-naming not yet accurate — see note above)
    'Ortho_Chlorine': 'c(Cl)c',
    'Ortho_Fluorine': 'c(F)c',
    'Meta_Fluorine': 'c(F)cc',
    'Para_Fluorine': 'c(F)ccc',
    'CF3_Group': 'C(F)(F)F',

    # Alkyl Chains
    'Butyl_Chain': 'CCCC',
    'Propyl_Chain': 'CCC',
    'Isopropyl': 'C(C)C',
    'tert_Butyl': 'C(C)(C)C',
}

pharmacophore_patterns = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in PHARMACOPHORE_SMARTS.items()
    if Chem.MolFromSmarts(smarts) is not None
}

# ------------------------------------------------------------------
# 2. Extraction functions
# ------------------------------------------------------------------
def extract_bemis_murcko(smiles):
    """Generates the generic Bemis-Murcko scaffold for a given SMILES string."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return 'Invalid_SMILES'
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        if not scaffold_smiles:
            return 'Acyclic/Linear'
        return scaffold_smiles
    except Exception:
        return 'Extraction_Error'


def extract_pharmacophores(smiles):
    """Checks the SMILES against predefined pharmacophore SMARTS and lists them."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return 'None'
    found_features = [
        name for name, pattern in pharmacophore_patterns.items()
        if mol.HasSubstructMatch(pattern)
    ]
    return " | ".join(found_features) if found_features else 'None'


# ------------------------------------------------------------------
# 3. Load and label
# ------------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv('fusion_inhibition_data.csv', low_memory=False)

columns_to_keep = [
    'PUBCHEM_SID',
    'PUBCHEM_EXT_DATASOURCE_SMILES',
    'PUBCHEM_ACTIVITY_OUTCOME',
    'WT_Inhibition @ 10 uM Avg',
]
df_simple = df[columns_to_keep].copy()
df_simple.columns = ['pubchem_id', 'smiles', 'activity_outcome', 'wt_average']

df_simple['wt_average'] = pd.to_numeric(df_simple['wt_average'], errors='coerce')

THRESHOLD = 57.94
# Compound must meet the quantitative threshold AND survive PubChem's curation
df_simple['active'] = (
    (df_simple['wt_average'] >= THRESHOLD) & (df_simple['activity_outcome'] == 'Active')
).astype(int)

# Drop rows with a missing SMILES string or missing numeric readout
df_simple = df_simple.dropna(subset=['smiles', 'wt_average'])

# ------------------------------------------------------------------
# 4. RDKit annotation (Bemis-Murcko + pharmacophores)
# ------------------------------------------------------------------
print("Extracting Bemis-Murcko scaffolds (this may take a few minutes)...")
df_simple['bm_scaffold'] = df_simple['smiles'].apply(extract_bemis_murcko)

print("Extracting pharmacophore features...")
df_simple['pharmacophore_features'] = df_simple['smiles'].apply(extract_pharmacophores)

n_before_struct_filter = len(df_simple)
df_simple = df_simple[
    ~df_simple['bm_scaffold'].isin(['Invalid_SMILES', 'Extraction_Error'])
].copy()
n_removed_unparseable = n_before_struct_filter - len(df_simple)
print(f"Removed {n_removed_unparseable} compounds with unparseable SMILES.")

# ------------------------------------------------------------------
# 5. Save and report
# ------------------------------------------------------------------
df_simple.to_csv('clean_dataset_bm_annotated.csv', index=False)

print("\n=== CLEANED & ANNOTATED DATASET ===")
print(f"Total compounds: {len(df_simple)}")
print(f"Active (Curated Hit): {df_simple['active'].sum()}")
print(f"Inactive: {(df_simple['active'] == 0).sum()}")
print(f"Active percentage: {df_simple['active'].sum() / len(df_simple) * 100:.2f}%")

print("\n=== TOP 10 BEMIS-MURCKO SCAFFOLDS ===")
top_scaffolds = (
    df_simple[df_simple['bm_scaffold'] != 'Acyclic/Linear']['bm_scaffold']
    .value_counts()
    .head(10)
)
print(top_scaffolds)

print("\nFirst 5 rows (showing new columns):")
print(df_simple[['pubchem_id', 'smiles', 'active', 'bm_scaffold', 'pharmacophore_features']].head())

print("\nLoading cleaned dataset...")
df_clean = pd.read_csv('clean_dataset_bm_annotated.csv', low_memory=False)

# =====================================================================
# Stage 3: Merge with Confirmatory Screen and Exclude Confirmed Compounds
# =====================================================================
print("\nLoading confirmatory screen (AID 1361) data table...")
confirmatory = pd.read_csv(
    'AID_1361_datatable.csv', skiprows=[1, 2, 3, 4], low_memory=False
)

confirmatory_ids = set(confirmatory['PUBCHEM_SID'].dropna())
print(f"Confirmatory screen compounds: {len(confirmatory)}")
print(f"Unique confirmatory SIDs: {len(confirmatory_ids)}")

# Reload the fully cleaned & annotated primary dataset produced in Stage 1
df_clean = pd.read_csv('clean_dataset_bm_annotated.csv', low_memory=False)

# Identify which primary-screen compounds also underwent confirmatory
# dose-response testing in AID 1361. These are set aside entirely -- they
# must NOT appear in training or testing, so they can later serve as a
# clean, unseen external validation set graded against real IC50/selectivity
# data rather than the noisier primary-screen label.
is_confirmatory_matched = df_clean['pubchem_id'].isin(confirmatory_ids)
confirmatory_holdout = df_clean[is_confirmatory_matched].copy()
final_dataset_without_confirmatory = df_clean[~is_confirmatory_matched].copy()

print(f"\nTotal cleaned compounds: {len(df_clean)}")
print(f"Matched into confirmatory holdout (removed from train/test pool): {len(confirmatory_holdout)}")
print(f"Remaining for training/testing: {len(final_dataset_without_confirmatory)}")

print("\nConfirmatory holdout label counts (primary-screen 'active' flag, for reference):")
print(confirmatory_holdout['active'].value_counts())

print("\nRemaining pool label counts:")
remaining_counts = final_dataset_without_confirmatory['active'].value_counts()
print(remaining_counts)
n_act = (final_dataset_without_confirmatory['active'] == 1).sum()
n_inact = (final_dataset_without_confirmatory['active'] == 0).sum()
print(f"Remaining pool imbalance ratio: {n_inact / n_act:.2f}:1")

# ------------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------------
overlap = set(final_dataset_without_confirmatory['pubchem_id']) & set(confirmatory_holdout['pubchem_id'])
assert len(overlap) == 0, "Leakage detected: some compounds appear in both sets!"
assert len(final_dataset_without_confirmatory) + len(confirmatory_holdout) == len(df_clean)
print(f"\nSanity check passed: no compound overlap, and counts sum to the original {len(df_clean)}.")

# ------------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------------
final_dataset_without_confirmatory.to_csv('final_dataset_without_confirmatory_data.csv', index=False)
confirmatory_holdout.to_csv('confirmatory_holdout_set.csv', index=False)

print("\nSaved 'final_dataset_without_confirmatory_data.csv' "
      f"({len(final_dataset_without_confirmatory)} compounds) -- use this for training/testing.")
print("Saved 'confirmatory_holdout_set.csv' "
      f"({len(confirmatory_holdout)} compounds) -- untouched external validation set.")

