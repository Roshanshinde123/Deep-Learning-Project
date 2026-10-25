import os
import json
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.data_loader import STATIC_PARAMS, DYNAMIC_PARAMS

# Biological plausibility bounds for outlier filtering
VALID_RANGES = {
    'HR': (20.0, 250.0),           # bpm
    'Temp': (25.0, 43.0),          # Celsius
    'SysABP': (30.0, 300.0),       # mmHg
    'DiasABP': (10.0, 200.0),      # mmHg
    'MAP': (20.0, 250.0),          # mmHg
    'NISysABP': (30.0, 300.0),     # mmHg
    'NIDiasABP': (10.0, 200.0),    # mmHg
    'NIMAP': (20.0, 250.0),        # mmHg
    'RespRate': (4.0, 80.0),       # breaths/min
    'GCS': (3.0, 15.0),            # score [3, 15]
    'pH': (6.7, 7.8),              # arterial pH
    'PaO2': (15.0, 600.0),         # mmHg
    'PaCO2': (10.0, 150.0),        # mmHg
    'SaO2': (50.0, 100.0),         # %
    'FiO2': (0.21, 1.0),           # fraction
    'Urine': (0.0, 5000.0),        # mL (0 is physiologically valid for anuria)
    'BUN': (1.0, 250.0),           # mg/dL
    'Creatinine': (0.1, 30.0),     # mg/dL
    'Glucose': (10.0, 1500.0),     # mg/dL
    'K': (1.0, 12.0),              # mEq/L
    'Na': (90.0, 185.0),           # mEq/L
    'WBC': (0.1, 200.0),           # x10^3/uL
    'Platelets': (2.0, 2000.0),    # x10^3/uL
    'HCT': (5.0, 75.0),            # %
    'Lactate': (0.1, 35.0),        # mmol/L
    'Bilirubin': (0.0, 90.0),      # mg/dL (0 is normal/valid)
    'AST': (1.0, 40000.0),         # U/L
    'ALT': (1.0, 20000.0),         # U/L
    'ALP': (5.0, 5000.0),          # U/L
    'Albumin': (0.5, 6.5),         # g/dL
    'TroponinT': (0.0, 35.0),      # ng/mL (0 is normal)
    'TroponinI': (0.0, 60.0),      # ng/mL (0 is normal)
    'Cholesterol': (10.0, 600.0),  # mg/dL
    'Mg': (0.1, 15.0),             # mg/dL
    'HCO3': (2.0, 60.0),           # mEq/L
    'MechVent': (0.0, 1.0),        # binary indicator
    'Weight': (20.0, 450.0),       # kg (dynamic weight)
    # Static descriptors
    'Weight_admission': (20.0, 450.0), # kg (admission weight)
    'Height': (50.0, 250.0),           # cm
    'Age': (15.0, 100.0),              # years
}

def clean_observation(param, val, cleaning_log):
    """
    Validates measurement against biological bounds.
    Converts out-of-range values and sensor artifacts to NaN.
    Preserves true zero measurements where clinically valid (e.g. Urine, Troponin, Bilirubin).
    """
    if np.isnan(val):
        return np.nan
        
    if param in VALID_RANGES:
        low, high = VALID_RANGES[param]
        if val < low or val > high:
            cleaning_log['outliers_filtered'][param] += 1
            return np.nan
            
    return val

def get_hour_bin(minutes):
    """
    Maps elapsed minutes since ICU admission into discrete hourly bins [0, 47].
    
    Binning Convention:
    - Hour 0:  [00:00, 01:00) -> 0 <= m < 60
    - Hour 1:  [01:00, 02:00) -> 60 <= m < 120
    - ...
    - Hour 22: [22:00, 23:00) -> 1320 <= m < 1380
    - Hour 23: [23:00, 24:00] -> 1380 <= m <= 1440 (Includes exactly 24:00)
    - Hour 24: (24:00, 25:00) -> 1440 < m < 1500
    - ...
    - Hour 46: [46:00, 47:00) -> 2760 <= m < 2820
    - Hour 47: [47:00, 48:00] -> 2820 <= m <= 2880 (Includes exactly 48:00)
    
    Guarantees:
    - Hours 0..23 strictly contain observations with m <= 1440 (no information > 24h).
    - Hours 0..47 strictly contain observations with m <= 2880 (no information > 48h).
    - Any observation with m < 0 or m > 2880 is invalid and dropped (returns None).
    """
    if minutes < 0 or minutes > 2880:
        return None
    if minutes == 2880:
        return 47
    if minutes == 1440:
        return 23
    return minutes // 60

def build_hourly_patient_grid(patient_record, cleaning_log):
    """
    Converts irregular observations of a single patient into a regular 48 x 37 grid.
    
    Continuous features: mean aggregation.
    Binary feature (MechVent): max aggregation (1 if ventilated during that hour).
    
    Returns:
        grid: np.ndarray of shape (48, 37) with NaNs for unobserved hours.
        mask: np.ndarray of shape (48, 37) with 1 if observed, 0 if missing.
    """
    num_features = len(DYNAMIC_PARAMS)
    param_to_idx = {p: i for i, p in enumerate(DYNAMIC_PARAMS)}
    
    hour_buckets = defaultdict(list)
    
    for minutes, param, val in patient_record['dynamic']:
        hour = get_hour_bin(minutes)
        if hour is None:
            cleaning_log['invalid_timestamps_dropped'] += 1
            continue
            
        clean_val = clean_observation(param, val, cleaning_log)
        if np.isnan(clean_val):
            continue
            
        p_idx = param_to_idx[param]
        hour_buckets[(hour, p_idx)].append(clean_val)
        
    grid = np.full((48, num_features), np.nan, dtype=np.float32)
    mask = np.zeros((48, num_features), dtype=np.float32)
    
    for (hour, p_idx), val_list in hour_buckets.items():
        param_name = DYNAMIC_PARAMS[p_idx]
        if param_name == 'MechVent':
            grid[hour, p_idx] = float(np.max(val_list))
        else:
            grid[hour, p_idx] = float(np.mean(val_list))
        mask[hour, p_idx] = 1.0
        
    return grid, mask

def extract_static_record(patient_record, cleaning_log, max_minutes=2880):
    """
    Extracts static admission descriptors as a dictionary:
    Age, Gender, Height, ICUType, Weight_admission, and MechVent_ever_recorded.
    
    max_minutes: 2880 for 48-hour horizon, 1440 for 24-hour horizon.
    """
    res = {}
    for p in STATIC_PARAMS:
        val = patient_record['static'].get(p, np.nan)
        clean_val = clean_observation(p, val, cleaning_log)
        res[p] = clean_val
        
    # Patient-level MechVent indicator within the specified horizon
    ever_mv = 0.0
    for minutes, param, val in patient_record['dynamic']:
        if param == 'MechVent' and 0 <= minutes <= max_minutes and val == 1.0:
            ever_mv = 1.0
            break
    res['MechVent_ever_recorded'] = ever_mv
    return res

def create_stratified_split(outcomes_a, random_state=42):
    """
    Partition Set A (4,000 patients) into 3,200 Train (80%) and 800 Validation (20%)
    using stratified splitting based on In-hospital_death.
    """
    train_df, val_df = train_test_split(
        outcomes_a,
        test_size=0.2,
        stratify=outcomes_a['In-hospital_death'],
        random_state=random_state
    )
    return set(train_df['RecordID']), set(val_df['RecordID'])

def compute_training_imputation_stats(train_grids, train_static_records):
    """
    Computes medians and modes strictly from Training Set (3,200 patients) only.
    No validation or test data is used.
    
    Returns:
        dynamic_medians: np.ndarray of shape (37,)
        static_impute_dict: dict of {param: impute_value}
    """
    # 1. Dynamic medians
    flat_train = train_grids.reshape(-1, train_grids.shape[-1])
    dynamic_medians = np.zeros(train_grids.shape[-1], dtype=np.float32)
    
    for f in range(train_grids.shape[-1]):
        param_name = DYNAMIC_PARAMS[f]
        if param_name == 'MechVent':
            # MechVent unrecorded reference level is 0.0
            dynamic_medians[f] = 0.0
        else:
            col_vals = flat_train[:, f]
            valid_vals = col_vals[~np.isnan(col_vals)]
            if len(valid_vals) > 0:
                dynamic_medians[f] = float(np.median(valid_vals))
            else:
                dynamic_medians[f] = 0.0
                
    # 2. Static impute values
    static_impute_dict = {}
    df_train_static = pd.DataFrame(train_static_records)
    for p in STATIC_PARAMS:
        valid_vals = df_train_static[p].dropna()
        if p in ['Gender', 'ICUType']:
            # Categorical mode
            static_impute_dict[p] = float(valid_vals.mode()[0])
        else:
            # Continuous median
            static_impute_dict[p] = float(valid_vals.median())
            
    return dynamic_medians, static_impute_dict

def impute_simple(grids, dynamic_medians):
    """
    Method A: Simple Imputation
    Replaces missing values (NaN) with training-set population medians.
    For MechVent: unobserved hours are filled with 0.0 (unrecorded baseline indicator with mask=0.0).
    """
    imputed = grids.copy()
    num_features = grids.shape[-1]
    for f in range(num_features):
        nan_mask = np.isnan(imputed[:, :, f])
        imputed[:, :, f][nan_mask] = dynamic_medians[f]
    return imputed

def impute_forward_fill(grids, dynamic_medians):
    """
    Method B: Historical Forward-Fill Imputation (LOCF)
    Carries previous available observations forward strictly along the time axis (hours 0 -> 47).
    Fills leading unobserved hours using training-set population medians.
    Strictly avoids backward-filling from future observations to prevent temporal leakage.
    Exempts MechVent: does not automatically forward-fill ventilation status across unrecorded hours.
    """
    N, T, F = grids.shape
    imputed = grids.copy()
    mechvent_idx = DYNAMIC_PARAMS.index('MechVent') if 'MechVent' in DYNAMIC_PARAMS else -1
    
    for i in range(N):
        for f in range(F):
            if f == mechvent_idx:
                # MechVent is an event record: do not forward fill across unobserved hours.
                # Unobserved hours remain 0.0 with mask 0.0.
                nan_mask = np.isnan(imputed[i, :, f])
                imputed[i, nan_mask, f] = 0.0
                continue
                
            last_known = np.nan
            for t in range(T):
                val = imputed[i, t, f]
                if not np.isnan(val):
                    last_known = val
                else:
                    if not np.isnan(last_known):
                        imputed[i, t, f] = last_known
                    else:
                        # Leading missing value before first measurement -> fill with train median
                        imputed[i, t, f] = dynamic_medians[f]
                        
    return imputed

def encode_and_scale_static_features(static_records_train, static_records_val, 
                                     static_records_test_b, static_records_test_c, 
                                     static_impute_dict):
    """
    Encodes and normalizes static admission descriptors without leakage.
    - Continuous features (Age, Height, Weight_admission): standardized using StandardScaler fit on Train.
    - Binary features (Gender, MechVent_ever_recorded): {0.0, 1.0}.
    - Categorical feature (ICUType): One-Hot Encoded into 4 binary indicators [CCU, CSRU, MedICU, SurgICU].
    
    Returns:
        S_train, S_val, S_test_b, S_test_c: np.ndarray of shape (N, 9)
        static_scaler: fitted StandardScaler for continuous static features
        static_feature_names: list of the 9 feature names
    """
    dfs = {
        'train': pd.DataFrame(static_records_train),
        'val': pd.DataFrame(static_records_val),
        'test_b': pd.DataFrame(static_records_test_b),
        'test_c': pd.DataFrame(static_records_test_c)
    }
    
    # 1. Impute missing static values using training statistics
    for part in dfs:
        for p in STATIC_PARAMS:
            dfs[part][p] = dfs[part][p].fillna(static_impute_dict[p])
            
    # Continuous static features
    cont_cols = ['Age', 'Height', 'Weight_admission']
    static_scaler = StandardScaler()
    static_scaler.fit(dfs['train'][cont_cols])
    
    static_arrays = {}
    for part in dfs:
        df = dfs[part]
        cont_scaled = static_scaler.transform(df[cont_cols]) # (N, 3)
        
        # Binary Gender & MechVent_ever_recorded
        gender_arr = df['Gender'].values.reshape(-1, 1).astype(np.float32) # (N, 1)
        mv_ever_arr = df['MechVent_ever_recorded'].values.reshape(-1, 1).astype(np.float32) # (N, 1)
        
        # One-Hot Encoded ICUType: 1=CCU, 2=CSRU, 3=MedICU, 4=SurgICU
        icu_arr = np.zeros((len(df), 4), dtype=np.float32)
        icu_types = df['ICUType'].values.astype(int)
        for idx, t in enumerate(icu_types):
            if 1 <= t <= 4:
                icu_arr[idx, t - 1] = 1.0
            else:
                # default to training mode if unexpected
                icu_arr[idx, int(static_impute_dict['ICUType']) - 1] = 1.0
                
        # Concatenate: (N, 3 cont + 1 gender + 1 mechvent_ever + 4 icu) = (N, 9)
        full_static = np.hstack([cont_scaled, gender_arr, mv_ever_arr, icu_arr]).astype(np.float32)
        static_arrays[part] = full_static
        
    static_names = [
        'Age_scaled', 'Height_scaled', 'Weight_admission_scaled',
        'Gender', 'MechVent_ever_recorded', 
        'ICUType_CCU', 'ICUType_CSRU', 'ICUType_MedICU', 'ICUType_SurgICU'
    ]
    
    return (static_arrays['train'], static_arrays['val'], 
            static_arrays['test_b'], static_arrays['test_c'], 
            static_scaler, static_names)

def fit_and_apply_scalers(train_temporal, val_temporal, test_b_temporal, test_c_temporal):
    """
    Fits StandardScaler strictly on Training partition (3,200 patients) for CONTINUOUS features only.
    Crucial Rule: MechVent is a binary indicator and is NOT standardized with continuous z-score.
    MechVent remains strictly binary in {0.0, 1.0}.
    
    Returns:
        Scaled temporal tensors: shape (N, T, 37)
        temporal_scaler: fitted StandardScaler (fitted on the 36 continuous features)
    """
    N_tr, T, F = train_temporal.shape
    mechvent_idx = DYNAMIC_PARAMS.index('MechVent')
    cont_indices = [i for i in range(F) if i != mechvent_idx]
    
    # Extract continuous slices
    tr_cont = train_temporal[:, :, cont_indices].reshape(-1, len(cont_indices))
    temporal_scaler = StandardScaler()
    temporal_scaler.fit(tr_cont)
    
    def transform_tensor(tensor):
        N, T, _ = tensor.shape
        t_cont = tensor[:, :, cont_indices].reshape(-1, len(cont_indices))
        t_cont_scaled = temporal_scaler.transform(t_cont).reshape(N, T, len(cont_indices))
        
        # MechVent slice remains strictly unstandardized binary {0.0, 1.0}
        t_mv = tensor[:, :, [mechvent_idx]]
        
        # Reassemble original feature order
        out = np.zeros_like(tensor, dtype=np.float32)
        out[:, :, cont_indices] = t_cont_scaled
        out[:, :, [mechvent_idx]] = t_mv
        return out
        
    X_tr_s = transform_tensor(train_temporal)
    X_val_s = transform_tensor(val_temporal)
    X_tb_s = transform_tensor(test_b_temporal)
    X_tc_s = transform_tensor(test_c_temporal)
    
    return X_tr_s, X_val_s, X_tb_s, X_tc_s, temporal_scaler
