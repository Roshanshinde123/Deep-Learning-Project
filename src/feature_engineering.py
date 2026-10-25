import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import numpy as np
import pandas as pd
from src.data_loader import DYNAMIC_PARAMS

def extract_patient_tabular_features(X, mask, static, dynamic_param_names=None, static_param_names=None):
    """
    Extracts comprehensive, non-redundant patient-level tabular features from 
    temporal arrays (values + observation masks) and static admission features.
    
    Parameters:
        X: np.ndarray of shape (N, T, F) - standardized continuous values + binary MechVent
        mask: np.ndarray of shape (N, T, F) - binary observation indicators (1 if measured, 0 if missing)
        static: np.ndarray of shape (N, S) - encoded static admission features (S=9)
        dynamic_param_names: list of length F parameter names (default: DYNAMIC_PARAMS)
        static_param_names: list of length S static feature names
        
    Returns:
        X_tabular: np.ndarray of shape (N, num_tabular_features)
        feature_names: list of string feature names
    """
    if dynamic_param_names is None:
        dynamic_param_names = DYNAMIC_PARAMS
        
    if static_param_names is None:
        static_param_names = [
            'Age_scaled', 'Height_scaled', 'Weight_admission_scaled',
            'Gender', 'MechVent_ever_recorded', 
            'ICUType_CCU', 'ICUType_CSRU', 'ICUType_MedICU', 'ICUType_SurgICU'
        ]
        
    N, T, F = X.shape
    feature_cols = []
    feature_names = []
    
    weight_idx = dynamic_param_names.index('Weight') if 'Weight' in dynamic_param_names else -1
    mechvent_idx = dynamic_param_names.index('MechVent') if 'MechVent' in dynamic_param_names else -1
    
    # -------------------------------------------------------------
    # 1. PROCESS DYNAMIC CLINICAL MEASUREMENTS
    # -------------------------------------------------------------
    for f in range(F):
        param = dynamic_param_names[f]
        M_f = mask[:, :, f] # (N, T)
        X_f = X[:, :, f]    # (N, T)
        
        counts = np.sum(M_f, axis=1) # (N,)
        pcts = counts / float(T)     # (N,)
        
        # ---------------------------------------------------------
        # SPECIAL CASE A: WEIGHT (AVOID REDUNDANCY WITH ADMISSION)
        # ---------------------------------------------------------
        if f == weight_idx:
            # First and last observed indices
            first_idx = np.argmax(M_f == 1.0, axis=1)
            first_vals = np.where(counts > 0, X_f[np.arange(N), first_idx], X_f[:, 0])
            last_idx = T - 1 - np.argmax(M_f[:, ::-1] == 1.0, axis=1)
            last_vals = np.where(counts > 0, X_f[np.arange(N), last_idx], X_f[:, 0])
            
            min_vals = np.where(counts > 0, np.min(np.where(M_f == 1.0, X_f, np.inf), axis=1), X_f[:, 0])
            max_vals = np.where(counts > 0, np.max(np.where(M_f == 1.0, X_f, -np.inf), axis=1), X_f[:, 0])
            
            # Dynamic features for Weight (avoid duplicating static admission weight)
            w_net_change = np.where(counts > 0, last_vals - first_vals, 0.0)
            w_variability = np.where(counts > 0, max_vals - min_vals, 0.0)
            dt_w = np.maximum(last_idx - first_idx, 1)
            w_trend = np.where((counts > 1) & (last_idx > first_idx), (last_vals - first_vals) / dt_w, 0.0)
            w_post_adm = np.any((M_f == 1.0) & (np.arange(T)[None, :] > 0), axis=1).astype(np.float32)
            
            feature_cols.extend([w_net_change, w_variability, w_trend, counts, pcts, w_post_adm])
            feature_names.extend([
                'Weight_net_change', 'Weight_variability', 'Weight_trend',
                'Weight_obs_count', 'Weight_obs_pct', 'Weight_has_post_admission'
            ])
            continue
            
        # ---------------------------------------------------------
        # SPECIAL CASE B: MECHANICAL VENTILATION (BINARY EVENT RECORD)
        # ---------------------------------------------------------
        if f == mechvent_idx:
            first_idx = np.argmax(M_f == 1.0, axis=1)
            mv_first = np.where(counts > 0, X_f[np.arange(N), first_idx], 0.0)
            last_idx = T - 1 - np.argmax(M_f[:, ::-1] == 1.0, axis=1)
            mv_last = np.where(counts > 0, X_f[np.arange(N), last_idx], 0.0)
            
            feature_cols.extend([counts, pcts, mv_first, mv_last])
            feature_names.extend([
                'MechVent_obs_count', 'MechVent_obs_pct', 'MechVent_first', 'MechVent_last'
            ])
            continue
            
        # ---------------------------------------------------------
        # GENERAL CONTINUOUS CLINICAL MEASUREMENTS (35 VARIABLES)
        # ---------------------------------------------------------
        # 1. Mean (computed on observed values if observed, otherwise imputed baseline)
        sum_vals = np.sum(X_f * M_f, axis=1)
        mean_vals = np.where(counts > 0, sum_vals / np.maximum(counts, 1.0), X_f[:, 0])
        
        # 2. Min & 3. Max
        min_vals = np.where(counts > 0, np.min(np.where(M_f == 1.0, X_f, np.inf), axis=1), X_f[:, 0])
        max_vals = np.where(counts > 0, np.max(np.where(M_f == 1.0, X_f, -np.inf), axis=1), X_f[:, 0])
        
        # 4. Std (sample std on observed measurements, 0.0 if <= 1 observation)
        diff_sq = ((X_f - mean_vals[:, None]) * M_f) ** 2
        std_vals = np.where(counts > 1, np.sqrt(np.sum(diff_sq, axis=1) / np.maximum(counts - 1, 1.0)), 0.0)
        
        # 5. First observed value & 6. Last observed value
        first_idx = np.argmax(M_f == 1.0, axis=1)
        first_vals = np.where(counts > 0, X_f[np.arange(N), first_idx], X_f[:, 0])
        last_idx = T - 1 - np.argmax(M_f[:, ::-1] == 1.0, axis=1)
        last_vals = np.where(counts > 0, X_f[np.arange(N), last_idx], X_f[:, 0])
        
        # 7. Count (counts) & 8. Percentage (pcts)
        
        # 9. Trend / Slope (rate of change between first and last observed values)
        dt = np.maximum(last_idx - first_idx, 1)
        trend_vals = np.where((counts > 1) & (last_idx > first_idx), (last_vals - first_vals) / dt, 0.0)
        
        feature_cols.extend([
            mean_vals, min_vals, max_vals, std_vals, 
            first_vals, last_vals, counts, pcts, trend_vals
        ])
        feature_names.extend([
            f'{param}_mean', f'{param}_min', f'{param}_max', f'{param}_std',
            f'{param}_first', f'{param}_last', f'{param}_obs_count', f'{param}_obs_pct', f'{param}_trend'
        ])
        
    # -------------------------------------------------------------
    # 2. PROCESS STATIC ADMISSION DESCRIPTORS (9 FEATURES)
    # -------------------------------------------------------------
    for s_idx in range(static.shape[1]):
        feature_cols.append(static[:, s_idx])
        feature_names.append(static_param_names[s_idx])
        
    # Concatenate all tabular columns
    X_tabular = np.column_stack(feature_cols).astype(np.float32)
    
    # Check for NaNs or Infs
    assert not np.isnan(X_tabular).any(), "NaNs detected in extracted tabular features!"
    assert not np.isinf(X_tabular).any(), "Infs detected in extracted tabular features!"
    
    return X_tabular, feature_names

def generate_all_tabular_datasets():
    """
    Builds and persists tabular feature sets across all experimental dimensions:
    - Imputation: Method A (Simple) vs Method B (Forward-Fill)
    - Prediction Horizon: 48 Hours vs 24 Hours
    - Partitions: Train (3,200), Val (800), Test B (4,000), Test C (4,000)
    
    Saves outputs to data/processed/tabular/.
    """
    out_dir = 'data/processed/tabular'
    os.makedirs(out_dir, exist_ok=True)
    
    scenarios = [
        ('forward_fill', '48h', 48, False),
        ('forward_fill', '24h', 24, True),
        ('simple_imputation', '48h', 48, False),
        ('simple_imputation', '24h', 24, True),
    ]
    
    results = {}
    
    for imp_method, horizon_name, T, is_24h in scenarios:
        src_dir = os.path.join('data/processed', imp_method)
        print(f"\nExtracting tabular features for [{imp_method.upper()} | {horizon_name}]...")
        
        # Load temporal arrays, masks, and static features
        t_suffix = '_24' if is_24h else '_48'
        s_suffix = '_24' if is_24h else ''
        
        X_tr = np.load(os.path.join(src_dir, f'X_train{t_suffix}.npy'))
        X_val = np.load(os.path.join(src_dir, f'X_val{t_suffix}.npy'))
        X_tb = np.load(os.path.join(src_dir, f'X_test_b{t_suffix}.npy'))
        X_tc = np.load(os.path.join(src_dir, f'X_test_c{t_suffix}.npy'))
        
        M_tr = np.load(os.path.join(src_dir, f'mask_train{t_suffix}.npy'))
        M_val = np.load(os.path.join(src_dir, f'mask_val{t_suffix}.npy'))
        M_tb = np.load(os.path.join(src_dir, f'mask_test_b{t_suffix}.npy'))
        M_tc = np.load(os.path.join(src_dir, f'mask_test_c{t_suffix}.npy'))
        
        S_tr = np.load(os.path.join(src_dir, f'static_train{s_suffix}.npy'))
        S_val = np.load(os.path.join(src_dir, f'static_val{s_suffix}.npy'))
        S_tb = np.load(os.path.join(src_dir, f'static_test_b{s_suffix}.npy'))
        S_tc = np.load(os.path.join(src_dir, f'static_test_c{s_suffix}.npy'))
        
        y_tr = np.load(os.path.join(src_dir, 'y_train.npy'))
        y_val = np.load(os.path.join(src_dir, 'y_val.npy'))
        y_tb = np.load(os.path.join(src_dir, 'y_test_b.npy'))
        y_tc = np.load(os.path.join(src_dir, 'y_test_c.npy'))
        
        # Feature extraction
        X_tr_tab, feat_names = extract_patient_tabular_features(X_tr, M_tr, S_tr)
        X_val_tab, _ = extract_patient_tabular_features(X_val, M_val, S_val)
        X_tb_tab, _ = extract_patient_tabular_features(X_tb, M_tb, S_tb)
        X_tc_tab, _ = extract_patient_tabular_features(X_tc, M_tc, S_tc)
        
        print(f"  Train Tabular:  {X_tr_tab.shape}")
        print(f"  Val Tabular:    {X_val_tab.shape}")
        print(f"  Test B Tabular: {X_tb_tab.shape}")
        print(f"  Test C Tabular: {X_tc_tab.shape}")
        
        # Save as npz archive
        save_path = os.path.join(out_dir, f'tabular_{imp_method}_{horizon_name}.npz')
        np.savez_compressed(
            save_path,
            X_train=X_tr_tab, y_train=y_tr,
            X_val=X_val_tab, y_val=y_val,
            X_test_b=X_tb_tab, y_test_b=y_tb,
            X_test_c=X_tc_tab, y_test_c=y_tc,
            feature_names=np.array(feat_names)
        )
        print(f"  Saved: {save_path}")
        
        results[f'{imp_method}_{horizon_name}'] = {
            'num_features': len(feat_names),
            'feature_names': feat_names,
            'train_shape': list(X_tr_tab.shape),
            'val_shape': list(X_val_tab.shape),
            'test_b_shape': list(X_tb_tab.shape),
            'test_c_shape': list(X_tc_tab.shape)
        }
        
    # Save feature names list
    with open(os.path.join(out_dir, 'tabular_feature_metadata.json'), 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\nAll tabular datasets successfully generated and saved to data/processed/tabular/!")
    return results

if __name__ == '__main__':
    generate_all_tabular_datasets()
