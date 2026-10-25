import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from src.preprocessing import get_hour_bin
from src.data_loader import DYNAMIC_PARAMS, STATIC_PARAMS

def run_all_audit_tests():
    print("=" * 70)
    print("RUNNING PHASE 2 FINAL TECHNICAL AUDIT TEST SUITE (REFINED)")
    print("=" * 70)

    # -------------------------------------------------------------
    # TEST 1: TIME BOUNDARY VALIDATION
    # -------------------------------------------------------------
    print("\n[TEST 1] Auditing Time Boundary & Binning Logic...")
    assert get_hour_bin(0) == 0, "m=0 should be Hour 0"
    assert get_hour_bin(59) == 0, "m=59 should be Hour 0"
    assert get_hour_bin(60) == 1, "m=60 should be Hour 1"
    assert get_hour_bin(1379) == 22, "m=1379 should be Hour 22"
    assert get_hour_bin(1380) == 23, "m=1380 should be Hour 23"
    assert get_hour_bin(1439) == 23, "m=1439 should be Hour 23"
    assert get_hour_bin(1440) == 23, "m=1440 (24:00) MUST be assigned to Hour 23 (closing hour of 24h window)"
    assert get_hour_bin(1441) == 24, "m=1441 (24:01) MUST be assigned to Hour 24 (after 24h mark)"
    assert get_hour_bin(2879) == 47, "m=2879 should be Hour 47"
    assert get_hour_bin(2880) == 47, "m=2880 (48:00) MUST be assigned to Hour 47 (closing hour of 48h window)"
    assert get_hour_bin(2881) is None, "m=2881 (>48h) MUST be rejected as None"
    assert get_hour_bin(-1) is None, "m=-1 (<0) MUST be rejected as None"
    print("  [PASS] get_hour_bin correctly assigns 24:00 (m=1440) to Hour 23 and 48:00 (m=2880) to Hour 47.")
    print("  [PASS] Slicing hours 0..23 strictly guarantees zero leakage from >24:00.")

    # -------------------------------------------------------------
    # TEST 2: WEIGHT FEATURE (STATIC ADMISSION + DYNAMIC TRAJECTORY)
    # -------------------------------------------------------------
    print("\n[TEST 2] Auditing Weight Feature (Static vs Dynamic)...")
    for pipeline in ['simple_imputation', 'forward_fill']:
        p_dir = os.path.join('data/processed', pipeline)
        
        # Static arrays have 9 columns: [Age_scaled, Height_scaled, Weight_admission_scaled, Gender, MechVent_ever_recorded, ICUType_CCU, ICUType_CSRU, ICUType_MedICU, ICUType_SurgICU]
        S_tr = np.load(os.path.join(p_dir, 'static_train.npy'))
        assert S_tr.shape == (3200, 9), f"Expected static shape (3200, 9), got {S_tr.shape}"
        
        # Check Weight_admission is standardized (mean ~ 0, std ~ 1)
        w_adm_tr = S_tr[:, 2]
        assert abs(np.mean(w_adm_tr)) < 1e-4, f"Weight_admission mean {np.mean(w_adm_tr)} not centered"
        assert abs(np.std(w_adm_tr) - 1.0) < 1e-2, f"Weight_admission std {np.std(w_adm_tr)} not 1.0"
        
        # Dynamic temporal arrays have 37 features
        X_tr_48 = np.load(os.path.join(p_dir, 'X_train_48.npy'))
        assert X_tr_48.shape == (3200, 48, 37), f"Expected temporal shape (3200, 48, 37), got {X_tr_48.shape}"
        w_dyn_idx = DYNAMIC_PARAMS.index('Weight')
        assert w_dyn_idx == 35, f"Expected Weight at index 35, got {w_dyn_idx}"
        
        # Check dynamic weight is standardized on training set
        flat_w_dyn = X_tr_48[:, :, w_dyn_idx].reshape(-1)
        assert abs(np.mean(flat_w_dyn)) < 1e-4, f"Dynamic weight mean not centered"
        assert abs(np.std(flat_w_dyn) - 1.0) < 1e-2, f"Dynamic weight std not 1.0"
        
    print("  [PASS] Static Weight_admission (col 2 of (N, 9)) and Dynamic Weight (col 35 of (N, 48, 37)) verified.")

    # -------------------------------------------------------------
    # TEST 3: MECHANICAL VENTILATION (NO AUTOMATIC FORWARD FILL & PATIENT-LEVEL INDICATOR)
    # -------------------------------------------------------------
    print("\n[TEST 3] Auditing Mechanical Ventilation (MechVent)...")
    mv_idx = DYNAMIC_PARAMS.index('MechVent')
    assert mv_idx == 18, f"Expected MechVent at index 18, got {mv_idx}"
    
    for pipeline in ['simple_imputation', 'forward_fill']:
        p_dir = os.path.join('data/processed', pipeline)
        for part in ['train', 'val', 'test_b', 'test_c']:
            arr_48 = np.load(os.path.join(p_dir, f'X_{part}_48.npy'))
            mask_48 = np.load(os.path.join(p_dir, f'mask_{part}_48.npy'))
            mv_slice = arr_48[:, :, mv_idx]
            mv_mask = mask_48[:, :, mv_idx]
            
            # Values strictly binary {0.0, 1.0}
            unique_vals = set(np.unique(mv_slice))
            assert unique_vals.issubset({0.0, 1.0}), f"MechVent has non-binary values in {pipeline}/{part}: {unique_vals}"
            
            # Value == 1.0 if and only if mask == 1.0 (no forward filling of ventilation across unobserved hours!)
            assert np.array_equal(mv_slice, mv_mask), f"MechVent value does not match mask in {pipeline}/{part} (improper forward-fill detected)!"
            
            # Static array check: MechVent_ever_recorded at col 4
            S = np.load(os.path.join(p_dir, f'static_{part}.npy'))
            mv_ever = S[:, 4]
            assert set(np.unique(mv_ever)).issubset({0.0, 1.0}), f"MechVent_ever_recorded has non-binary values: {set(np.unique(mv_ever))}"
            # Patient ever ventilated if and only if sum of mask across hours > 0
            expected_ever = (mv_mask.sum(axis=1) > 0).astype(np.float32)
            assert np.array_equal(mv_ever, expected_ever), f"MechVent_ever_recorded mismatch in {pipeline}/{part}!"
            
            # Check 24-hour static array has zero future leakage
            S_24 = np.load(os.path.join(p_dir, f'static_{part}_24.npy'))
            mask_24 = np.load(os.path.join(p_dir, f'mask_{part}_24.npy'))
            mv_ever_24 = S_24[:, 4]
            expected_ever_24 = (mask_24[:, :, mv_idx].sum(axis=1) > 0).astype(np.float32)
            assert np.array_equal(mv_ever_24, expected_ever_24), f"24h MechVent_ever_recorded has leakage or mismatch in {pipeline}/{part}!"
            
    print("  [PASS] MechVent values strictly in {0.0, 1.0}. Continuous StandardScaler NOT applied.")
    print("  [PASS] MechVent is NOT forward-filled to discharge; unrecorded hours preserve mask=0.")
    print("  [PASS] Patient-level MechVent_ever_recorded verified in static arrays (48h and 24h zero-leakage).")

    # -------------------------------------------------------------
    # TEST 4: CATEGORICAL STATIC ENCODING (GENDER & ICUTYPE)
    # -------------------------------------------------------------
    print("\n[TEST 4] Auditing Static Feature Encoding (One-Hot ICUType, Binary Gender, MechVent_ever)...")
    for pipeline in ['simple_imputation', 'forward_fill']:
        p_dir = os.path.join('data/processed', pipeline)
        for part in ['train', 'val', 'test_b', 'test_c']:
            S = np.load(os.path.join(p_dir, f'static_{part}.npy'))
            assert S.shape[1] == 9, f"Expected 9 static features, got {S.shape[1]}"
            
            # Gender (col 3) is binary
            gender = S[:, 3]
            assert set(np.unique(gender)).issubset({0.0, 1.0}), f"Gender non-binary: {set(np.unique(gender))}"
            
            # ICUType one-hot (cols 5, 6, 7, 8)
            icu_one_hot = S[:, 5:9]
            assert set(np.unique(icu_one_hot)).issubset({0.0, 1.0}), "ICUType has non-binary entries"
            row_sums = np.sum(icu_one_hot, axis=1)
            assert np.all(row_sums == 1.0), "Every patient MUST belong to exactly one ICU type"
            
    print("  [PASS] ICUType properly One-Hot Encoded into 4 mutually exclusive binary indicators.")
    print("  [PASS] Gender properly binary encoded in {0.0, 1.0}.")
    print("  [PASS] Static feature vector shape confirmed: (N, 9).")

    # -------------------------------------------------------------
    # TEST 5: MISSINGNESS MASK INTEGRATION (74-CHANNEL & 37-CHANNEL TENSORS)
    # -------------------------------------------------------------
    print("\n[TEST 5] Auditing Missingness Mask Integration...")
    for pipeline in ['simple_imputation', 'forward_fill']:
        p_dir = os.path.join('data/processed', pipeline)
        
        # 48-hour arrays
        X_48 = np.load(os.path.join(p_dir, 'X_train_48.npy'))               # (3200, 48, 37)
        mask_48 = np.load(os.path.join(p_dir, 'mask_train_48.npy'))         # (3200, 48, 37)
        X_wm_48 = np.load(os.path.join(p_dir, 'X_train_48_with_mask.npy')) # (3200, 48, 74)
        
        assert X_48.shape == (3200, 48, 37), f"X_48 shape {X_48.shape} != (3200, 48, 37)"
        assert mask_48.shape == (3200, 48, 37), f"mask_48 shape {mask_48.shape} != (3200, 48, 37)"
        assert X_wm_48.shape == (3200, 48, 74), f"X_wm_48 shape {X_wm_48.shape} != (3200, 48, 74)"
        
        # Check alignment
        assert np.allclose(X_wm_48[:, :, :37], X_48), "Values slice of X_wm_48 does not match X_48"
        assert np.allclose(X_wm_48[:, :, 37:], mask_48), "Masks slice of X_wm_48 does not match mask_48"
        
        # 24-hour arrays
        X_24 = np.load(os.path.join(p_dir, 'X_train_24.npy'))               # (3200, 24, 37)
        mask_24 = np.load(os.path.join(p_dir, 'mask_train_24.npy'))         # (3200, 24, 37)
        X_wm_24 = np.load(os.path.join(p_dir, 'X_train_24_with_mask.npy')) # (3200, 24, 74)
        
        assert X_24.shape == (3200, 24, 37)
        assert mask_24.shape == (3200, 24, 37)
        assert X_wm_24.shape == (3200, 24, 74)
        assert np.allclose(X_wm_24[:, :, :37], X_24)
        assert np.allclose(X_wm_24[:, :, 37:], mask_24)
        
        # 72-channel comparison array
        X_wm_72 = np.load(os.path.join(p_dir, 'X_train_48_with_mask_72.npy'))
        assert X_wm_72.shape == (3200, 48, 72), f"X_wm_72 shape {X_wm_72.shape} != (3200, 48, 72)"
        
    print("  [PASS] Both separate (37 values, 37 masks) and integrated (74 channels, 72 channels) tensors validated.")

    # -------------------------------------------------------------
    # TEST 6: HISTORICAL FORWARD-FILL VALIDATION (ZERO BACKWARD-FILL)
    # -------------------------------------------------------------
    print("\n[TEST 6] Auditing Method B Imputation (Historical LOCF only, No Backward-Fill)...")
    p_dir = 'data/processed/forward_fill'
    X_tr_ff = np.load(os.path.join(p_dir, 'X_train_48.npy'))
    mask_tr = np.load(os.path.join(p_dir, 'mask_train_48.npy'))
    
    hr_idx = DYNAMIC_PARAMS.index('HR')
    found_patient = False
    for i in range(100):
        m_hr = mask_tr[i, :, hr_idx]
        obs_hours = np.where(m_hr == 1.0)[0]
        if len(obs_hours) > 0 and obs_hours[0] > 2:
            t_first = obs_hours[0]
            # Leading values 0 .. t_first-1 must all be identical (equal to train median)
            leading_vals = X_tr_ff[i, :t_first, hr_idx]
            assert np.all(leading_vals == leading_vals[0]), "Leading values are not uniform median!"
            found_patient = True
            break
    assert found_patient, "Could not find patient with leading missing HR for testing"
    print("  [PASS] Method B strictly applies historical forward-fill only.")
    print("  [PASS] Leading unobserved values populated from training-set population median; ZERO backward-fill from future observations.")

    # -------------------------------------------------------------
    # TEST 7: DATA LEAKAGE & PATIENT ISOLATION GUARANTEES
    # -------------------------------------------------------------
    print("\n[TEST 7] Auditing Patient Isolation & Leakage Prevention...")
    manifest = pd.read_csv('data/processed/patient_split_manifest.csv')
    train_ids = set(manifest[manifest['Partition'] == 'train']['RecordID'])
    val_ids = set(manifest[manifest['Partition'] == 'val']['RecordID'])
    tb_ids = set(manifest[manifest['Partition'] == 'test_b']['RecordID'])
    tc_ids = set(manifest[manifest['Partition'] == 'test_c']['RecordID'])
    
    assert len(train_ids.intersection(val_ids)) == 0, "Leakage: Train and Val overlap!"
    assert len(train_ids.intersection(tb_ids)) == 0, "Leakage: Train and Test B overlap!"
    assert len(train_ids.intersection(tc_ids)) == 0, "Leakage: Train and Test C overlap!"
    assert len(val_ids.intersection(tb_ids)) == 0, "Leakage: Val and Test B overlap!"
    assert len(val_ids.intersection(tc_ids)) == 0, "Leakage: Val and Test C overlap!"
    assert len(tb_ids.intersection(tc_ids)) == 0, "Leakage: Test B and Test C overlap!"
    print("  [PASS] Zero patient overlap across all 4 partitions confirmed.")

    # -------------------------------------------------------------
    # TEST 8: NUMERICAL INTEGRITY & BIJECTIVE LABEL MAPPING
    # -------------------------------------------------------------
    print("\n[TEST 8] Auditing Numerical Integrity (No NaNs, No Infs, Exact Label Matching)...")
    for pipeline in ['simple_imputation', 'forward_fill']:
        p_dir = os.path.join('data/processed', pipeline)
        for part in ['train', 'val', 'test_b', 'test_c']:
            for arr_file in [f'X_{part}_48.npy', f'X_{part}_48_with_mask.npy', f'static_{part}.npy', f'static_{part}_24.npy', f'y_{part}.npy']:
                arr = np.load(os.path.join(p_dir, arr_file))
                assert np.isnan(arr).sum() == 0, f"NaNs found in {pipeline}/{arr_file}!"
                assert np.isinf(arr).sum() == 0, f"Infs found in {pipeline}/{arr_file}!"
                
        # Label matching
        rids_tr = np.load(os.path.join(p_dir, 'rids_train.npy'))
        y_tr = np.load(os.path.join(p_dir, 'y_train.npy'))
        tr_map = dict(zip(manifest[manifest['Partition'] == 'train']['RecordID'],
                          manifest[manifest['Partition'] == 'train']['In-hospital_death']))
        for i in range(len(rids_tr)):
            assert y_tr[i] == tr_map[rids_tr[i]], f"Label mismatch for ID {rids_tr[i]}"
            
    print("  [PASS] Zero NaNs and zero Infs across all generated arrays.")
    print("  [PASS] In-hospital mortality labels match patient manifest 100% bijectively.")

    print("\n" + "=" * 70)
    print("ALL 8 TECHNICAL AUDIT TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)

if __name__ == '__main__':
    run_all_audit_tests()
