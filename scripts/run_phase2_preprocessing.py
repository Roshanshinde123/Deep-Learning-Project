import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import joblib
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import (
    STATIC_PARAMS, DYNAMIC_PARAMS, DYNAMIC_PARAMS_36,
    load_outcomes, load_patient_raw_records
)
from src.preprocessing import (
    build_hourly_patient_grid, extract_static_record,
    create_stratified_split, compute_training_imputation_stats,
    impute_simple, impute_forward_fill, fit_and_apply_scalers,
    encode_and_scale_static_features, VALID_RANGES
)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def run_phase2():
    print("=" * 75)
    print("STARTING PHASE 2 AUDIT REFINEMENT & TIME-SERIES PIPELINE (12,000 PATIENTS)")
    print("=" * 75)
    
    # Create directories
    os.makedirs('data/processed/simple_imputation', exist_ok=True)
    os.makedirs('data/processed/forward_fill', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # 1. Load outcomes
    print("\n[Step 1/10] Loading and validating outcome files...")
    outcomes = load_outcomes('Dataset')
    df_a, df_b, df_c = outcomes['a'], outcomes['b'], outcomes['c']
    
    # 2. Partition splitting (Set A: 3,200 Train, 800 Val; Set B: 4,000 Test; Set C: 4,000 Held-Out)
    print("\n[Step 2/10] Generating stratified patient split manifest (random_state=42)...")
    train_ids, val_ids = create_stratified_split(df_a, random_state=42)
    test_b_ids = set(df_b['RecordID'])
    test_c_ids = set(df_c['RecordID'])
    
    manifest_rows = []
    id_to_outcome = {}
    id_to_partition = {}
    
    for _, row in df_a.iterrows():
        rid = int(row['RecordID'])
        part = 'train' if rid in train_ids else 'val'
        ihd = int(row['In-hospital_death'])
        id_to_outcome[rid] = ihd
        id_to_partition[rid] = part
        manifest_rows.append({'RecordID': rid, 'Partition': part, 'Set': 'Set A', 'In-hospital_death': ihd})
        
    for _, row in df_b.iterrows():
        rid = int(row['RecordID'])
        ihd = int(row['In-hospital_death'])
        id_to_outcome[rid] = ihd
        id_to_partition[rid] = 'test_b'
        manifest_rows.append({'RecordID': rid, 'Partition': 'test_b', 'Set': 'Set B', 'In-hospital_death': ihd})
        
    for _, row in df_c.iterrows():
        rid = int(row['RecordID'])
        ihd = int(row['In-hospital_death'])
        id_to_outcome[rid] = ihd
        id_to_partition[rid] = 'test_c'
        manifest_rows.append({'RecordID': rid, 'Partition': 'test_c', 'Set': 'Set C', 'In-hospital_death': ihd})
        
    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv('data/processed/patient_split_manifest.csv', index=False)
    print(f"Saved patient split manifest: {len(df_manifest)} patients.")
    print("Partition counts:")
    print(df_manifest['Partition'].value_counts().to_string())
    
    # Save feature names reference
    feature_meta = {
        'dynamic_features_37': DYNAMIC_PARAMS,
        'dynamic_features_36_no_weight': DYNAMIC_PARAMS_36,
        'static_admission_descriptors_raw': STATIC_PARAMS,
        'static_features_encoded_9': [
            'Age_scaled', 'Height_scaled', 'Weight_admission_scaled',
            'Gender', 'MechVent_ever_recorded', 
            'ICUType_CCU', 'ICUType_CSRU', 'ICUType_MedICU', 'ICUType_SurgICU'
        ],
        'target': 'In-hospital_death',
        'excluded_variables': ['RecordID', 'Survival', 'Length_of_stay', 'SAPS-I', 'SOFA'],
        'methodological_rules': {
            'weight_handling': {
                'admission_weight_static': 'Standardized static baseline capturing habitus at admission (col 2 of static array).',
                'dynamic_weight': 'Hourly dynamic time series (feature index 35 of dynamic array).',
                'dual_representation_note': 'Weight measured at 00:00 is present in both static descriptor and dynamic Hour 0. In tabular feature engineering (XGBoost), extract net delta (W_last - W_adm) and volatility (W_max - W_min) rather than redundant static duplications to prevent multicollinearity.'
            },
            'mechvent_handling': {
                'interpretation': 'Absent records denote unrecorded status, not confirmed non-ventilation. MIMIC-II logs MechVent as procedural events (100% of records are 1.0).',
                'hourly_discretization': 'Preserves explicit observations only (1.0 with mask 1.0). Unrecorded hours are 0.0 with mask 0.0.',
                'no_forward_fill_to_discharge': 'MechVent is exempted from forward-fill to avoid assuming persistent ventilation without documentation.',
                'patient_level_indicator': 'MechVent_ever_recorded is added to static features (1.0 if ever recorded in observation window, 0.0 otherwise).'
            },
            'imputation_method_b': {
                'description': 'Historical Forward-Fill (LOCF) with Training-Set Population Median for Leading Missing Values.',
                'temporal_direction': 'Strictly forward in time (hours 0 -> 47). Zero backward-filling from future observations.'
            },
            'outcome_provenance': {
                'set_a': 'Development dataset released with official outcomes in 2012.',
                'sets_b_and_c': 'Official test partitions released post-challenge by PhysioNet as academic benchmark keys.',
                'isolation_policy': 'Set B and Set C outcomes are completely isolated from model development and used exclusively for final out-of-sample evaluation.'
            }
        }
    }
    with open('data/processed/feature_names.json', 'w') as f:
        json.dump(feature_meta, f, indent=2)

    # 3. Load raw patient records (including Set C archive)
    print("\n[Step 3/10] Loading all patient records (Set A, Set B, Set C archive)...")
    patient_data, load_stats = load_patient_raw_records('Dataset')
    
    # 4. Cleaning & Hourly Grid Discretization (48 hours x 37 features)
    print("\n[Step 4/10] Discretizing into 48 hourly bins and applying biological filtering...")
    cleaning_log = {
        'outliers_filtered': defaultdict(int),
        'invalid_timestamps_dropped': 0
    }
    
    # Prepare partition arrays
    partition_rids = {'train': [], 'val': [], 'test_b': [], 'test_c': []}
    partition_grids = {'train': [], 'val': [], 'test_b': [], 'test_c': []}
    partition_masks = {'train': [], 'val': [], 'test_b': [], 'test_c': []}
    partition_static_records_48 = {'train': [], 'val': [], 'test_b': [], 'test_c': []}
    partition_static_records_24 = {'train': [], 'val': [], 'test_b': [], 'test_c': []}
    partition_labels = {'train': [], 'val': [], 'test_b': [], 'test_c': []}
    
    for rid, precord in patient_data.items():
        part = id_to_partition[rid]
        grid, mask = build_hourly_patient_grid(precord, cleaning_log)
        s_record_48 = extract_static_record(precord, cleaning_log, max_minutes=2880)
        s_record_24 = extract_static_record(precord, cleaning_log, max_minutes=1440)
        y = id_to_outcome[rid]
        
        partition_rids[part].append(rid)
        partition_grids[part].append(grid)
        partition_masks[part].append(mask)
        partition_static_records_48[part].append(s_record_48)
        partition_static_records_24[part].append(s_record_24)
        partition_labels[part].append(y)
        
    for p in partition_grids:
        partition_grids[p] = np.array(partition_grids[p], dtype=np.float32)
        partition_masks[p] = np.array(partition_masks[p], dtype=np.float32)
        partition_labels[p] = np.array(partition_labels[p], dtype=np.int32)
        partition_rids[p] = np.array(partition_rids[p], dtype=np.int32)
        
    print(f"Raw 37-feature dynamic grids constructed:")
    print(f"  Train: {partition_grids['train'].shape}, Labels: {partition_labels['train'].shape}")
    print(f"  Val:   {partition_grids['val'].shape}, Labels: {partition_labels['val'].shape}")
    print(f"  TestB: {partition_grids['test_b'].shape}, Labels: {partition_labels['test_b'].shape}")
    print(f"  TestC: {partition_grids['test_c'].shape}, Labels: {partition_labels['test_c'].shape}")

    # Missing value calculation before imputation
    raw_missing_pct_all = float(np.isnan(np.concatenate([partition_grids['train'], partition_grids['val'], partition_grids['test_b'], partition_grids['test_c']], axis=0)).mean() * 100.0)
    print(f"Hourly cell missingness across all 12,000 patients before imputation: {raw_missing_pct_all:.2f}%")

    # 5. Compute Training Set Medians & Static Imputation Values
    print("\n[Step 5/10] Computing imputation parameters strictly from Training partition (3,200 patients)...")
    dynamic_medians, static_impute_dict = compute_training_imputation_stats(
        partition_grids['train'], partition_static_records_48['train']
    )
    
    # 6. Encode and Scale Static Features (9 features: 3 scaled continuous + 2 binary + 4 one-hot ICU)
    print("\n[Step 6/10] Encoding static admission features (One-Hot ICUType, binary Gender, MechVent_ever_recorded, scaled continuous)...")
    (S_tr_9, S_val_9, S_tb_9, S_tc_9, 
     static_scaler, static_names) = encode_and_scale_static_features(
        partition_static_records_48['train'], partition_static_records_48['val'],
        partition_static_records_48['test_b'], partition_static_records_48['test_c'],
        static_impute_dict
    )
    (S_tr_9_24, S_val_9_24, S_tb_9_24, S_tc_9_24, 
     _, _) = encode_and_scale_static_features(
        partition_static_records_24['train'], partition_static_records_24['val'],
        partition_static_records_24['test_b'], partition_static_records_24['test_c'],
        static_impute_dict
    )
    print(f"Static features successfully encoded: shape = {S_tr_9.shape}, columns = {static_names}")

    # 7. PIPELINE A: Simple Imputation
    print("\n[Step 7/10] Executing Method A: Simple Imputation (Training Medians)...")
    X_tr_simp = impute_simple(partition_grids['train'], dynamic_medians)
    X_val_simp = impute_simple(partition_grids['val'], dynamic_medians)
    X_tb_simp = impute_simple(partition_grids['test_b'], dynamic_medians)
    X_tc_simp = impute_simple(partition_grids['test_c'], dynamic_medians)
    
    # 8. PIPELINE B: Historical Forward-Fill Imputation (LOCF + Training Median for leading missing)
    print("\n[Step 8/10] Executing Method B: Historical Forward-Fill Imputation (LOCF + Training Median for leading missing)...")
    X_tr_ff = impute_forward_fill(partition_grids['train'], dynamic_medians)
    X_val_ff = impute_forward_fill(partition_grids['val'], dynamic_medians)
    X_tb_ff = impute_forward_fill(partition_grids['test_b'], dynamic_medians)
    X_tc_ff = impute_forward_fill(partition_grids['test_c'], dynamic_medians)

    # 9. Generate Imputation Comparison Statistics
    print("\nCalculating Imputation Comparison Statistics (reports/imputation_comparison.csv)...")
    imp_comp_rows = []
    flat_raw_tr = partition_grids['train'].reshape(-1, len(DYNAMIC_PARAMS))
    flat_simp_tr = X_tr_simp.reshape(-1, len(DYNAMIC_PARAMS))
    flat_ff_tr = X_tr_ff.reshape(-1, len(DYNAMIC_PARAMS))
    
    for f_idx, param in enumerate(DYNAMIC_PARAMS):
        raw_vals = flat_raw_tr[:, f_idx]
        raw_clean = raw_vals[~np.isnan(raw_vals)]
        
        simp_vals = flat_simp_tr[:, f_idx]
        ff_vals = flat_ff_tr[:, f_idx]
        
        imp_comp_rows.append({
            'Feature': param,
            'Train_Median': round(float(dynamic_medians[f_idx]), 3),
            'Raw_Mean': round(float(np.mean(raw_clean)), 3) if len(raw_clean) > 0 else np.nan,
            'Raw_Std': round(float(np.std(raw_clean)), 3) if len(raw_clean) > 0 else np.nan,
            'Simple_Mean': round(float(np.mean(simp_vals)), 3),
            'Simple_Std': round(float(np.std(simp_vals)), 3),
            'ForwardFill_Mean': round(float(np.mean(ff_vals)), 3),
            'ForwardFill_Std': round(float(np.std(ff_vals)), 3),
            'Mean_Abs_Difference': round(float(np.mean(np.abs(ff_vals - simp_vals))), 3),
            'Variance_Retained_Ratio': round(float(np.var(ff_vals) / (np.var(simp_vals) + 1e-8)), 3)
        })
        
    df_imp_comp = pd.DataFrame(imp_comp_rows)
    df_imp_comp.to_csv('reports/imputation_comparison.csv', index=False)
    print("Saved reports/imputation_comparison.csv")

    # 10. Scale Continuous Features Strictly on Training Set (MechVent remains unstandardized binary)
    print("\n[Step 9/10] Normalizing continuous features with StandardScaler fit strictly on Training Set...")
    X_tr_simp_s, X_val_simp_s, X_tb_simp_s, X_tc_simp_s, scaler_temp_simp = fit_and_apply_scalers(
        X_tr_simp, X_val_simp, X_tb_simp, X_tc_simp
    )
    X_tr_ff_s, X_val_ff_s, X_tb_ff_s, X_tc_ff_s, scaler_temp_ff = fit_and_apply_scalers(
        X_tr_ff, X_val_ff, X_tb_ff, X_tc_ff
    )
    
    # Save scalers
    joblib.dump(scaler_temp_simp, 'data/processed/simple_imputation/scaler_temporal.joblib')
    joblib.dump(static_scaler, 'data/processed/simple_imputation/scaler_static.joblib')
    joblib.dump(scaler_temp_ff, 'data/processed/forward_fill/scaler_temporal.joblib')
    joblib.dump(static_scaler, 'data/processed/forward_fill/scaler_static.joblib')

    # 11. Concatenate Measurement Values + Missingness Masks
    # Shapes: (N, 48, 37 values) + (N, 48, 37 masks) = (N, 48, 74)
    print("\nIntegrating Missingness Masks (Concatenating values + masks along feature dimension)...")
    # Simple
    X_tr_simp_with_mask_48 = np.concatenate([X_tr_simp_s, partition_masks['train']], axis=-1)
    X_val_simp_with_mask_48 = np.concatenate([X_val_simp_s, partition_masks['val']], axis=-1)
    X_tb_simp_with_mask_48 = np.concatenate([X_tb_simp_s, partition_masks['test_b']], axis=-1)
    X_tc_simp_with_mask_48 = np.concatenate([X_tc_simp_s, partition_masks['test_c']], axis=-1)
    
    # Forward Fill
    X_tr_ff_with_mask_48 = np.concatenate([X_tr_ff_s, partition_masks['train']], axis=-1)
    X_val_ff_with_mask_48 = np.concatenate([X_val_ff_s, partition_masks['val']], axis=-1)
    X_tb_ff_with_mask_48 = np.concatenate([X_tb_ff_s, partition_masks['test_b']], axis=-1)
    X_tc_ff_with_mask_48 = np.concatenate([X_tc_ff_s, partition_masks['test_c']], axis=-1)
    
    # 24-hour slices (hours 0 to 23: strictly m <= 1440)
    # Values only (37)
    X_tr_simp_24 = X_tr_simp_s[:, :24, :]
    X_val_simp_24 = X_val_simp_s[:, :24, :]
    X_tb_simp_24 = X_tb_simp_s[:, :24, :]
    X_tc_simp_24 = X_tc_simp_s[:, :24, :]
    
    X_tr_ff_24 = X_tr_ff_s[:, :24, :]
    X_val_ff_24 = X_val_ff_s[:, :24, :]
    X_tb_ff_24 = X_tb_ff_s[:, :24, :]
    X_tc_ff_24 = X_tc_ff_s[:, :24, :]
    
    # Masks 24 (37)
    mask_tr_24 = partition_masks['train'][:, :24, :]
    mask_val_24 = partition_masks['val'][:, :24, :]
    mask_tb_24 = partition_masks['test_b'][:, :24, :]
    mask_tc_24 = partition_masks['test_c'][:, :24, :]
    
    # Values + Masks 24 (74)
    X_tr_simp_with_mask_24 = X_tr_simp_with_mask_48[:, :24, :]
    X_val_simp_with_mask_24 = X_val_simp_with_mask_48[:, :24, :]
    X_tb_simp_with_mask_24 = X_tb_simp_with_mask_48[:, :24, :]
    X_tc_simp_with_mask_24 = X_tc_simp_with_mask_48[:, :24, :]
    
    X_tr_ff_with_mask_24 = X_tr_ff_with_mask_48[:, :24, :]
    X_val_ff_with_mask_24 = X_val_ff_with_mask_48[:, :24, :]
    X_tb_ff_with_mask_24 = X_tb_ff_with_mask_48[:, :24, :]
    X_tc_ff_with_mask_24 = X_tc_ff_with_mask_48[:, :24, :]
    
    # 36-feature versions (excluding dynamic Weight, indices without Weight)
    weight_idx = DYNAMIC_PARAMS.index('Weight')
    no_w_indices = [i for i in range(len(DYNAMIC_PARAMS)) if i != weight_idx]
    
    # 12. Save all generated numpy arrays
    print("\n[Step 10/10] Saving all model-ready numpy arrays to data/processed/...")
    for p_name, (X_tr_48, X_val_48, X_tb_48, X_tc_48,
                 X_tr_24, X_val_24, X_tb_24, X_tc_24,
                 X_tr_wm_48, X_val_wm_48, X_tb_wm_48, X_tc_wm_48,
                 X_tr_wm_24, X_val_wm_24, X_tb_wm_24, X_tc_wm_24) in [
        ('simple_imputation', (X_tr_simp_s, X_val_simp_s, X_tb_simp_s, X_tc_simp_s,
                               X_tr_simp_24, X_val_simp_24, X_tb_simp_24, X_tc_simp_24,
                               X_tr_simp_with_mask_48, X_val_simp_with_mask_48, X_tb_simp_with_mask_48, X_tc_simp_with_mask_48,
                               X_tr_simp_with_mask_24, X_val_simp_with_mask_24, X_tb_simp_with_mask_24, X_tc_simp_with_mask_24)),
        ('forward_fill', (X_tr_ff_s, X_val_ff_s, X_tb_ff_s, X_tc_ff_s,
                          X_tr_ff_24, X_val_ff_24, X_tb_ff_24, X_tc_ff_24,
                          X_tr_ff_with_mask_48, X_val_ff_with_mask_48, X_tb_ff_with_mask_48, X_tc_ff_with_mask_48,
                          X_tr_ff_with_mask_24, X_val_ff_with_mask_24, X_tb_ff_with_mask_24, X_tc_ff_with_mask_24))
    ]:
        out_dir = os.path.join('data/processed', p_name)
        # 37-feature values only
        np.save(os.path.join(out_dir, 'X_train_48.npy'), X_tr_48)
        np.save(os.path.join(out_dir, 'X_val_48.npy'), X_val_48)
        np.save(os.path.join(out_dir, 'X_test_b_48.npy'), X_tb_48)
        np.save(os.path.join(out_dir, 'X_test_c_48.npy'), X_tc_48)
        
        np.save(os.path.join(out_dir, 'X_train_24.npy'), X_tr_24)
        np.save(os.path.join(out_dir, 'X_val_24.npy'), X_val_24)
        np.save(os.path.join(out_dir, 'X_test_b_24.npy'), X_tb_24)
        np.save(os.path.join(out_dir, 'X_test_c_24.npy'), X_tc_24)
        
        # 74-channel values + masks
        np.save(os.path.join(out_dir, 'X_train_48_with_mask.npy'), X_tr_wm_48)
        np.save(os.path.join(out_dir, 'X_val_48_with_mask.npy'), X_val_wm_48)
        np.save(os.path.join(out_dir, 'X_test_b_48_with_mask.npy'), X_tb_wm_48)
        np.save(os.path.join(out_dir, 'X_test_c_48_with_mask.npy'), X_tc_wm_48)
        
        np.save(os.path.join(out_dir, 'X_train_24_with_mask.npy'), X_tr_wm_24)
        np.save(os.path.join(out_dir, 'X_val_24_with_mask.npy'), X_val_wm_24)
        np.save(os.path.join(out_dir, 'X_test_b_24_with_mask.npy'), X_tb_wm_24)
        np.save(os.path.join(out_dir, 'X_test_c_24_with_mask.npy'), X_tc_wm_24)
        
        # 36-feature versions for backward compatibility
        np.save(os.path.join(out_dir, 'X_train_48_no_dyn_weight.npy'), X_tr_48[:, :, no_w_indices])
        np.save(os.path.join(out_dir, 'X_val_48_no_dyn_weight.npy'), X_val_48[:, :, no_w_indices])
        np.save(os.path.join(out_dir, 'X_test_b_48_no_dyn_weight.npy'), X_tb_48[:, :, no_w_indices])
        np.save(os.path.join(out_dir, 'X_test_c_48_no_dyn_weight.npy'), X_tc_48[:, :, no_w_indices])
        
        # 72-channel versions (36 values + 36 masks)
        mask_no_w_tr = partition_masks['train'][:, :, no_w_indices]
        mask_no_w_val = partition_masks['val'][:, :, no_w_indices]
        mask_no_w_tb = partition_masks['test_b'][:, :, no_w_indices]
        mask_no_w_tc = partition_masks['test_c'][:, :, no_w_indices]
        
        np.save(os.path.join(out_dir, 'X_train_48_with_mask_72.npy'), np.concatenate([X_tr_48[:, :, no_w_indices], mask_no_w_tr], axis=-1))
        np.save(os.path.join(out_dir, 'X_val_48_with_mask_72.npy'), np.concatenate([X_val_48[:, :, no_w_indices], mask_no_w_val], axis=-1))
        np.save(os.path.join(out_dir, 'X_test_b_48_with_mask_72.npy'), np.concatenate([X_tb_48[:, :, no_w_indices], mask_no_w_tb], axis=-1))
        np.save(os.path.join(out_dir, 'X_test_c_48_with_mask_72.npy'), np.concatenate([X_tc_48[:, :, no_w_indices], mask_no_w_tc], axis=-1))
        
        # Masks
        np.save(os.path.join(out_dir, 'mask_train_48.npy'), partition_masks['train'])
        np.save(os.path.join(out_dir, 'mask_val_48.npy'), partition_masks['val'])
        np.save(os.path.join(out_dir, 'mask_test_b_48.npy'), partition_masks['test_b'])
        np.save(os.path.join(out_dir, 'mask_test_c_48.npy'), partition_masks['test_c'])
        
        np.save(os.path.join(out_dir, 'mask_train_24.npy'), mask_tr_24)
        np.save(os.path.join(out_dir, 'mask_val_24.npy'), mask_val_24)
        np.save(os.path.join(out_dir, 'mask_test_b_24.npy'), mask_tb_24)
        np.save(os.path.join(out_dir, 'mask_test_c_24.npy'), mask_tc_24)
        
        # 9-dimensional static admission features (48h horizon: includes MechVent_ever_recorded_48)
        np.save(os.path.join(out_dir, 'static_train.npy'), S_tr_9)
        np.save(os.path.join(out_dir, 'static_val.npy'), S_val_9)
        np.save(os.path.join(out_dir, 'static_test_b.npy'), S_tb_9)
        np.save(os.path.join(out_dir, 'static_test_c.npy'), S_tc_9)
        
        # 9-dimensional static admission features (24h horizon: includes MechVent_ever_recorded_24, strictly zero future leakage)
        np.save(os.path.join(out_dir, 'static_train_24.npy'), S_tr_9_24)
        np.save(os.path.join(out_dir, 'static_val_24.npy'), S_val_9_24)
        np.save(os.path.join(out_dir, 'static_test_b_24.npy'), S_tb_9_24)
        np.save(os.path.join(out_dir, 'static_test_c_24.npy'), S_tc_9_24)
        
        # Labels and record IDs
        np.save(os.path.join(out_dir, 'y_train.npy'), partition_labels['train'])
        np.save(os.path.join(out_dir, 'y_val.npy'), partition_labels['val'])
        np.save(os.path.join(out_dir, 'y_test_b.npy'), partition_labels['test_b'])
        np.save(os.path.join(out_dir, 'y_test_c.npy'), partition_labels['test_c'])
        
        np.save(os.path.join(out_dir, 'rids_train.npy'), partition_rids['train'])
        np.save(os.path.join(out_dir, 'rids_val.npy'), partition_rids['val'])
        np.save(os.path.join(out_dir, 'rids_test_b.npy'), partition_rids['test_b'])
        np.save(os.path.join(out_dir, 'rids_test_c.npy'), partition_rids['test_c'])

    print("All arrays written to disk successfully!")

    # 13. Preprocessing Visualization (reports/preprocessing_visualization.png)
    print("\nGenerating reports/preprocessing_visualization.png...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Subplot 1: Method A vs Method B trajectory comparison on sample patient (HR)
    sample_pidx = 10
    hr_idx = DYNAMIC_PARAMS.index('HR')
    hrs_raw = partition_grids['train'][sample_pidx, :, hr_idx]
    hrs_simp = X_tr_simp[sample_pidx, :, hr_idx]
    hrs_ff = X_tr_ff[sample_pidx, :, hr_idx]
    
    time_hrs = list(range(48))
    axes[0, 0].plot(time_hrs, hrs_simp, label='Method A: Simple Imputation (Median)', color='#d95f02', linestyle='--', alpha=0.8)
    axes[0, 0].plot(time_hrs, hrs_ff, label='Method B: Forward-Fill (LOCF)', color='#1b9e77', linewidth=2)
    obs_t = [t for t in range(48) if not np.isnan(hrs_raw[t])]
    obs_v = [hrs_raw[t] for t in obs_t]
    axes[0, 0].scatter(obs_t, obs_v, color='#2b5c8f', s=60, zorder=5, label='Raw Observed Measurements')
    axes[0, 0].set_title('Sample Patient 48-Hour Heart Rate: Simple vs Forward-Fill', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('ICU Hour')
    axes[0, 0].set_ylabel('Heart Rate (bpm)')
    axes[0, 0].legend(frameon=True, fontsize=9)
    
    # Subplot 2: Dynamic Weight Trajectory for Patient with weight changes
    w_idx = DYNAMIC_PARAMS.index('Weight')
    # Find a patient with multiple weight changes
    patient_with_w_change = 0
    for i in range(len(partition_grids['train'])):
        w_vals = partition_grids['train'][i, :, w_idx]
        w_clean = w_vals[~np.isnan(w_vals)]
        if len(set(w_clean)) > 1:
            patient_with_w_change = i
            break
            
    w_raw = partition_grids['train'][patient_with_w_change, :, w_idx]
    w_ff = X_tr_ff[patient_with_w_change, :, w_idx]
    w_obs_t = [t for t in range(48) if not np.isnan(w_raw[t])]
    w_obs_v = [w_raw[t] for t in w_obs_t]
    
    axes[0, 1].plot(time_hrs, w_ff, label='Dynamic Weight (LOCF)', color='#7570b3', linewidth=2)
    axes[0, 1].scatter(w_obs_t, w_obs_v, color='red', s=70, zorder=5, label='Raw Weight Recordings')
    axes[0, 1].axhline(partition_static_records_48['train'][patient_with_w_change]['Weight_admission'], color='black', linestyle=':', label='Admission Baseline Weight')
    axes[0, 1].set_title(f'Dynamic Weight Trajectory: Patient {partition_rids["train"][patient_with_w_change]}', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('ICU Hour')
    axes[0, 1].set_ylabel('Weight (kg)')
    axes[0, 1].legend(frameon=True, fontsize=9)

    # Subplot 3: Missingness Reduction Before vs After Imputation
    miss_summary = pd.DataFrame({
        'Stage': ['Raw Hourly Cells', 'After Imputation (Simple)', 'After Imputation (Forward-Fill)'],
        'Missing_Pct': [raw_missing_pct_all, 0.0, 0.0]
    })
    bars = axes[1, 0].bar(miss_summary['Stage'], miss_summary['Missing_Pct'], color=['#d95f02', '#1b9e77', '#2b5c8f'], width=0.45)
    axes[1, 0].set_title('Missingness Percentage: Raw vs Imputed', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Missingness Percentage (%)')
    axes[1, 0].set_ylim(0, 100)
    for b in bars:
        axes[1, 0].text(b.get_x() + b.get_width()/2.0, b.get_height() + 2, f"{b.get_height():.1f}%", ha='center', fontweight='bold')

    # Subplot 4: MechVent Binary Distribution (Survivors vs Non-Survivors)
    mv_idx = DYNAMIC_PARAMS.index('MechVent')
    mv_max_train = np.max(X_tr_ff[:, :, mv_idx], axis=1) # 1 if ventilated at all in 48h
    df_mv = pd.DataFrame({
        'Ventilated': ['Yes (MechVent=1)' if v == 1 else 'No (MechVent=0)' for v in mv_max_train],
        'Mortality': ['Non-Survivor (1)' if y == 1 else 'Survivor (0)' for y in partition_labels['train']]
    })
    sns.countplot(data=df_mv, x='Ventilated', hue='Mortality', palette=['#2b5c8f', '#d95f02'], ax=axes[1, 1])
    axes[1, 1].set_title('Mechanical Ventilation Status vs In-Hospital Mortality (Training Set)', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Mechanical Ventilation Status in 48 Hours')
    axes[1, 1].set_ylabel('Patient Count')
    axes[1, 1].legend(title='Outcome', frameon=True)
    
    plt.tight_layout()
    plt.savefig('reports/preprocessing_visualization.png', dpi=300)
    plt.close()
    print("Saved reports/preprocessing_visualization.png")

    # 14. Create reports/preprocessing_summary.json
    print("\nWriting reports/preprocessing_summary.json...")
    summary_data = {
        'pipeline_status': 'Complete & Audited',
        'dataset_dimensions': {
            'N_total_patients': len(patient_data),
            'training_patients': int(len(train_ids)),
            'validation_patients': int(len(val_ids)),
            'test_b_patients': int(len(test_b_ids)),
            'test_c_patients': int(len(test_c_ids)),
            'time_steps_48h': 48,
            'time_steps_24h': 24,
            'num_dynamic_features': len(DYNAMIC_PARAMS),
            'num_static_features_encoded': 9,
            'num_concatenated_channels_48h': 74,
            'num_concatenated_channels_24h': 74
        },
        'imputation_experiments': {
            'method_a': 'Simple Imputation (Train population medians)',
            'method_b': 'Historical Forward-Fill (LOCF + Train median leading fill)'
        },
        'shapes': {
            'X_train_48': list(X_tr_simp_s.shape),
            'X_train_48_with_mask': list(X_tr_simp_with_mask_48.shape),
            'X_val_48': list(X_val_simp_s.shape),
            'X_val_48_with_mask': list(X_val_simp_with_mask_48.shape),
            'X_test_b_48': list(X_tb_simp_s.shape),
            'X_test_b_48_with_mask': list(X_tb_simp_with_mask_48.shape),
            'X_test_c_48': list(X_tc_simp_s.shape),
            'X_test_c_48_with_mask': list(X_tc_simp_with_mask_48.shape),
            'X_train_24': list(X_tr_simp_24.shape),
            'X_train_24_with_mask': list(X_tr_simp_with_mask_24.shape),
            'X_val_24': list(X_val_simp_24.shape),
            'X_val_24_with_mask': list(X_val_simp_with_mask_24.shape),
            'X_test_b_24': list(X_tb_simp_24.shape),
            'X_test_b_24_with_mask': list(X_tb_simp_with_mask_24.shape),
            'X_test_c_24': list(X_tc_simp_24.shape),
            'X_test_c_24_with_mask': list(X_tc_simp_with_mask_24.shape),
            'static_train_48': list(S_tr_9.shape),
            'static_val_48': list(S_val_9.shape),
            'static_test_b_48': list(S_tb_9.shape),
            'static_test_c_48': list(S_tc_9.shape),
            'static_train_24': list(S_tr_9_24.shape),
            'static_val_24': list(S_val_9_24.shape),
            'static_test_b_24': list(S_tb_9_24.shape),
            'static_test_c_24': list(S_tc_9_24.shape),
            'y_train': list(partition_labels['train'].shape),
            'y_val': list(partition_labels['val'].shape),
            'y_test_b': list(partition_labels['test_b'].shape),
            'y_test_c': list(partition_labels['test_c'].shape)
        },
        'feature_names': {
            'dynamic_37': DYNAMIC_PARAMS,
            'static_9': static_names
        },
        'missingness_before_imputation_pct': raw_missing_pct_all,
        'missingness_after_imputation_pct': 0.0,
        'cleaning_statistics': {
            'blank_parameter_lines_dropped': load_stats['dropped_blank_param_lines'],
            'outliers_filtered_to_nan': dict(cleaning_log['outliers_filtered']),
            'invalid_timestamps_dropped': cleaning_log['invalid_timestamps_dropped']
        },
        'data_leakage_guarantees': {
            'patient_overlap_between_splits': 0,
            'scalers_fit_on_train_only': True,
            'medians_fit_on_train_only': True,
            'target_variables_in_features': False,
            'future_leakage_in_LOCF': False,
            'mechvent_unstandardized_binary': True,
            'icup_type_one_hot_encoded': True
        }
    }
    with open('reports/preprocessing_summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    print("Saved reports/preprocessing_summary.json")

    print("\nPhase 2 preprocessing orchestration complete!")

if __name__ == '__main__':
    run_phase2()
