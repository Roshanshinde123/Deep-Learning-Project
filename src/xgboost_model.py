import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import itertools
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

def calculate_scale_pos_weight(y_train):
    """
    Computes scale_pos_weight strictly on training labels to address class imbalance.
    scale_pos_weight = count(negative) / count(positive)
    """
    y_train = np.asarray(y_train)
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    if pos_count == 0:
        return 1.0
    return float(neg_count) / float(pos_count)

def train_xgboost_model(X_train, y_train, X_val=None, y_val=None, params=None, 
                        early_stopping_rounds=30, verbose=False):
    """
    Trains an XGBoost binary classifier with early stopping and class weighting.
    
    Parameters:
        X_train, y_train: training features and binary labels
        X_val, y_val: validation features and binary labels for early stopping
        params: dict of XGBoost parameters
        early_stopping_rounds: int, rounds without validation improvement
        verbose: bool or int
        
    Returns:
        model: trained xgb.XGBClassifier
    """
    default_spw = calculate_scale_pos_weight(y_train)
    
    default_params = {
        'n_estimators': 800,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'scale_pos_weight': default_spw,
        'objective': 'binary:logistic',
        'eval_metric': ['aucpr', 'auc', 'logloss'],
        'random_state': 42,
        'tree_method': 'hist',
        'n_jobs': -1
    }
    
    if params:
        default_params.update(params)
        
    model = xgb.XGBClassifier(**default_params)
    
    if X_val is not None and y_val is not None:
        eval_set = [(X_train, y_train), (X_val, y_val)]
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=verbose
        )
    else:
        model.fit(X_train, y_train, verbose=verbose)
        
    return model

def tune_xgboost_hyperparameters(X_train, y_train, X_val, y_val, max_trials=16):
    """
    Systematically tunes XGBoost hyperparameters using Training and Validation sets only.
    Optimizes for Validation AUPRC (Average Precision) due to class imbalance.
    
    Parameters:
        X_train, y_train: training data
        X_val, y_val: validation data
        max_trials: int, maximum hyperparameter combinations to evaluate
        
    Returns:
        best_model: fitted XGBClassifier with optimal hyperparameters
        best_params: dict of best hyperparameters
        tuning_results: list of dicts summarizing all evaluated configurations
    """
    spw = calculate_scale_pos_weight(y_train)
    print(f"Computed Training scale_pos_weight: {spw:.3f} (Negatives: {np.sum(y_train==0)}, Positives: {np.sum(y_train==1)})")
    
    # Candidate grid tailored for tabular ICU physiology data
    candidate_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.03, 0.05, 0.08],
        'subsample': [0.7, 0.85],
        'colsample_bytree': [0.65, 0.8],
        'min_child_weight': [2, 5],
        'gamma': [0.1, 0.5]
    }
    
    keys = list(candidate_grid.keys())
    all_combos = list(itertools.product(*[candidate_grid[k] for k in keys]))
    
    # Deterministic selection of configurations
    rng = np.random.RandomState(42)
    selected_indices = rng.choice(len(all_combos), size=min(max_trials, len(all_combos)), replace=False)
    
    best_score = -1.0
    best_params = None
    best_model = None
    tuning_records = []
    
    print(f"Beginning systematic tuning across {len(selected_indices)} candidate configurations...")
    
    for trial_idx, idx in enumerate(selected_indices):
        combo = all_combos[idx]
        current_params = dict(zip(keys, combo))
        current_params.update({
            'n_estimators': 600,
            'scale_pos_weight': spw,
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'random_state': 42,
            'tree_method': 'hist',
            'n_jobs': -1,
            'early_stopping_rounds': 25
        })
        
        m = xgb.XGBClassifier(**current_params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        val_probs = m.predict_proba(X_val)[:, 1]
        val_auprc = float(average_precision_score(y_val, val_probs))
        val_auroc = float(roc_auc_score(y_val, val_probs))
        
        record = {
            'trial': trial_idx + 1,
            'params': {k: current_params[k] for k in keys},
            'best_iteration': int(m.best_iteration) if hasattr(m, 'best_iteration') and m.best_iteration is not None else 600,
            'val_auprc': round(val_auprc, 4),
            'val_auroc': round(val_auroc, 4)
        }
        tuning_records.append(record)
        print(f"  [Trial {trial_idx+1:02d}/{len(selected_indices)}] Depth={current_params['max_depth']}, LR={current_params['learning_rate']}, Sub={current_params['subsample']}, Col={current_params['colsample_bytree']} -> Val AUPRC: {val_auprc:.4f}, AUROC: {val_auroc:.4f}")
        
        if val_auprc > best_score:
            best_score = val_auprc
            best_params = current_params
            best_model = m
            
    print(f"\nOptimal Hyperparameters Selected (Best Val AUPRC = {best_score:.4f}):")
    for k in keys:
        print(f"  {k}: {best_params[k]}")
        
    return best_model, best_params, tuning_records

def save_xgboost_model(model, filepath='models/xgboost/best_xgboost_model.json'):
    """Saves fitted XGBoost model in native JSON format."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save_model(filepath)
    print(f"Saved fitted XGBoost model to {filepath}")

def load_xgboost_model(filepath='models/xgboost/best_xgboost_model.json'):
    """Loads fitted XGBoost model from native JSON format."""
    model = xgb.XGBClassifier()
    model.load_model(filepath)
    return model
