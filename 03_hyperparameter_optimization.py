"""
MitoFusionAI Pipeline — 03 Hyperparameter Optimization
Source notebook: optimize_model_with_optuna.ipynb
"""

# ==================== SUPPRESS WARNINGS ====================
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import optuna
import xgboost as xgb
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import json

print("""
╔══════════════════════════════════════════════════════════════════════╗
║             MITOFUSIONAI - OPTUNA BAYESIAN OPTIMIZATION              ║
║         Target Metric: PR-AUC (Precision-Recall Area Under Curve)    ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# ==================== 1. LOAD PRE-COMPUTED TRAINING DATA ====================
print("[1/3] Loading pre-computed explicit feature matrix...")

try:
    # Load the 91-feature matrix we saved during the last training run
    with open('training_features_matrix.pkl', 'rb') as f:
        X_train_matrix = pickle.load(f)
    
    # We need the training labels. We load the clean dataset and subset it using our saved indices.
    df = pd.read_csv('final_dataset_without_confirmatory_data.csv')
    train_idx = np.load('train_indices.npy')
    y_train = df['active'].values[train_idx]
    
    print(f"✓ Loaded {X_train_matrix.shape[0]} compounds with {X_train_matrix.shape[1]} features.")
    
    # Calculate baseline imbalance for reference
    imbalance_ratio = len(y_train[y_train==0]) / len(y_train[y_train==1])
    print(f"✓ Target Imbalance Ratio: {imbalance_ratio:.2f}:1")

except Exception as e:
    print(f"✗ ERROR loading data: {e}. Make sure training_features_matrix.pkl exists.")
    exit()

# ==================== 2. DEFINE THE OPTUNA OBJECTIVE ====================
def objective(trial):
    """
    The objective function is the 'arena' where Optuna tests different parameters.
    It returns the PR-AUC score, which Optuna will try to maximize.
    """
    # 1. Define the search space for hyperparameters
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        
        # Crucial for severe imbalance: Test different dampening factors for the weight
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 10.0, imbalance_ratio),
        
        # Prevent wild probability swings
        'max_delta_step': trial.suggest_int('max_delta_step', 1, 5),
        
        # Regularization (Feature Selection & Smoothing)
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        
        # Stochasticity (Prevents overfitting)
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        
        'tree_method': 'hist', # 'hist' is much faster for large datasets
        'random_state': 42,
        'verbosity': 0
    }

    # 2. Setup 3-Fold Stratified Cross Validation
    # We use StratifiedKFold to ensure every slice has actives in it (preventing crashes)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    pr_auc_scores = []

    # 3. Train and Evaluate
    for train_fold_idx, val_fold_idx in skf.split(X_train_matrix, y_train):
        X_tr, X_val = X_train_matrix[train_fold_idx], X_train_matrix[val_fold_idx]
        y_tr, y_val = y_train[train_fold_idx], y_train[val_fold_idx]

        model = xgb.XGBClassifier(**param)
        model.fit(X_tr, y_tr)
        
        # Predict on validation fold
        preds = model.predict_proba(X_val)[:, 1]
        
        # Calculate PR-AUC
        score = average_precision_score(y_val, preds)
        pr_auc_scores.append(score)

    # Optuna will attempt to maximize this mean score
    return np.mean(pr_auc_scores)

# ==================== 3. RUN THE STUDY ====================
print("\n[2/3] Initializing Optuna Bayesian Optimization...")
print("This will run 50 trials. Grab a coffee, this will take ~10-15 minutes.\n")

# Create a study object and specify the direction is 'maximize'
study = optuna.create_study(direction='maximize', study_name="MitoFusion_XGBoost_Opt")

# Run the optimization for 50 trials
study.optimize(objective, n_trials=50, show_progress_bar=True)

# ==================== 4. RESULTS AND SAVING ====================
print("\n[3/3] OPTIMIZATION COMPLETE!")
print("=" * 60)
print(f"Best PR-AUC Score achieved: {study.best_value:.4f}")
print("Best Hyperparameters Found:")

best_params = study.best_params
for key, value in best_params.items():
    if isinstance(value, float):
        print(f"  {key:20s}: {value:.4f}")
    else:
        print(f"  {key:20s}: {value}")

# Save the best parameters to a JSON file so they can be loaded into your training script
with open('optuna_best_params.json', 'w') as f:
    json.dump(best_params, f, indent=4)

print("\n✓ Best parameters saved to 'optuna_best_params.json'")
print("You can now update your training script with these exact parameters!")
