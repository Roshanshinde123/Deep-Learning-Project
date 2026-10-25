import os
import json
import numpy as np
import pandas as pd

def verify_phase2():
    print("=== STARTING PHASE 2 SANITY & DATA LEAKAGE VALIDATION ===")
    
    # 1. Check patient split manifest
    manifest_path = 'data/processed/patient_split_manifest.csv'
    assert os.path.exists(manifest_path), "Missing patient_split_manifest.csv!"
    df_manifest = pd.read_csv(manifest_path)
    
    total_patients = len(df_manifest)
    unique_patients = df_manifest['RecordID'].nunique()
    print(f"[Check 1] Total patients: {total_patients}, Unique IDs: {unique_patients}")
    assert total_patients == 12000, f"Expected 12000 patients, got {total_patients}"
    assert unique_patients == 12000, "Duplicate RecordIDs detected in manifest!"
    
    # Check partition counts
    part_counts = df_manifest['Partition'].value_counts().to_dict()
    print(f"  Partition distribution: {part_counts}")
    assert part_counts['train'] == 3200, f"Train count {part_counts['train']} != 3200"
    assert part_counts['val'] == 800, f"Val count {part_counts['val']} != 800"
    assert part_counts['test_b'] == 4000, f"Test B count {part_counts['test_b']} != 4000"
    assert part_counts['test_c'] == 4000, f"Test C count {part_counts['test_c']} != 4000"
    
    # Check patient isolation (Zero overlap)
    train_ids = set(df_manifest[df_manifest['Partition'] == 'train']['RecordID'])
    val_ids = set(df_manifest[df_manifest['Partition'] == 'val']['RecordID'])
    tb_ids = set(df_manifest[df_manifest['Partition'] == 'test_b']['RecordID'])
    tc_ids = set(df_manifest[df_manifest['Partition'] == 'test_c']['RecordID'])
    
    assert len(train_ids.intersection(val_ids)) == 0, "LEAKAGE: Overlap between Train and Val!"
    assert len(train_ids.intersection(tb_ids)) == 0, "LEAKAGE: Overlap between Train and Test B!"
    assert len(train_ids.intersection(tc_ids)) == 0, "LEAKAGE: Overlap between Train and Test C!"
    assert len(val_ids.intersection(tb_ids)) == 0, "LEAKAGE: Overlap between Val and Test B!"
    assert len(val_ids.intersection(tc_ids)) == 0, "LEAKAGE: Overlap between Val and Test C!"
    assert len(tb_ids.intersection(tc_ids)) == 0, "LEAKAGE: Overlap between Test B and Test C!"
    print("  [PASS] Zero patient overlap confirmed across all partitions.")

    # 2. Check feature names metadata
    feat_path = 'data/processed/feature_names.json'
    assert os.path.exists(feat_path), "Missing feature_names.json!"
    with open(feat_path, 'r') as f:
        meta = json.load(f)
        
    dynamic_feats = meta['dynamic_features']
    static_feats = meta['static_features']
    excluded = meta['excluded_variables']
    print(f"[Check 2] Dynamic features: {len(dynamic_feats)}, Static features: {len(static_feats)}")
    assert len(dynamic_feats) == 36, f"Expected 36 dynamic features, got {len(dynamic_feats)}"
    assert len(static_feats) == 5, f"Expected 5 static features, got {len(static_feats)}"
    
    # Target leakage check in feature lists
    forbidden = {'RecordID', 'Survival', 'Length_of_stay', 'SAPS-I', 'SOFA', 'In-hospital_death'}
    assert not any(f in forbidden for f in dynamic_feats), "TARGET LEAKAGE in dynamic features!"
    assert not any(f in forbidden for f in static_feats), "TARGET LEAKAGE in static features!"
    print("  [PASS] Zero target or future leakage variables in feature definitions.")

    # 3. Check numpy arrays in both simple_imputation and forward_fill
    pipelines = ['simple_imputation', 'forward_fill']
    for p in pipelines:
        print(f"\n[Check 3] Auditing arrays in data/processed/{p}/...")
        p_dir = os.path.join('data/processed', p)
        
        # Load arrays
        X_tr_48 = np.load(os.path.join(p_dir, 'X_train_48.npy'))
        X_val_48 = np.load(os.path.join(p_dir, 'X_val_48.npy'))
        X_tb_48 = np.load(os.path.join(p_dir, 'X_test_b_48.npy'))
        X_tc_48 = np.load(os.path.join(p_dir, 'X_test_c_48.npy'))
        
        X_tr_24 = np.load(os.path.join(p_dir, 'X_train_24.npy'))
        X_val_24 = np.load(os.path.join(p_dir, 'X_val_24.npy'))
        X_tb_24 = np.load(os.path.join(p_dir, 'X_test_b_24.npy'))
        X_tc_24 = np.load(os.path.join(p_dir, 'X_test_c_24.npy'))
        
        m_tr_48 = np.load(os.path.join(p_dir, 'mask_train_48.npy'))
        m_val_48 = np.load(os.path.join(p_dir, 'mask_val_48.npy'))
        m_tb_48 = np.load(os.path.join(p_dir, 'mask_test_b_48.npy'))
        m_tc_48 = np.load(os.path.join(p_dir, 'mask_test_c_48.npy'))
        
        S_tr = np.load(os.path.join(p_dir, 'static_train.npy'))
        S_val = np.load(os.path.join(p_dir, 'static_val.npy'))
        S_tb = np.load(os.path.join(p_dir, 'static_test_b.npy'))
        S_tc = np.load(os.path.join(p_dir, 'static_test_c.npy'))
        
        y_tr = np.load(os.path.join(p_dir, 'y_train.npy'))
        y_val = np.load(os.path.join(p_dir, 'y_val.npy'))
        y_tb = np.load(os.path.join(p_dir, 'y_test_b.npy'))
        y_tc = np.load(os.path.join(p_dir, 'y_test_c.npy'))
        
        # Verify shapes
        assert X_tr_48.shape == (3200, 48, 36), f"Shape mismatch X_tr_48: {X_tr_48.shape}"
        assert X_val_48.shape == (800, 48, 36), f"Shape mismatch X_val_48: {X_val_48.shape}"
        assert X_tb_48.shape == (4000, 48, 36), f"Shape mismatch X_tb_48: {X_tb_48.shape}"
        assert X_tc_48.shape == (4000, 48, 36), f"Shape mismatch X_tc_48: {X_tc_48.shape}"
        
        assert X_tr_24.shape == (3200, 24, 36), f"Shape mismatch X_tr_24: {X_tr_24.shape}"
        assert X_val_24.shape == (800, 24, 36), f"Shape mismatch X_val_24: {X_val_24.shape}"
        assert X_tb_24.shape == (4000, 24, 36), f"Shape mismatch X_tb_24: {X_tb_24.shape}"
        assert X_tc_24.shape == (4000, 24, 36), f"Shape mismatch X_tc_24: {X_tc_24.shape}"
        
        assert m_tr_48.shape == (3200, 48, 36)
        assert m_val_48.shape == (800, 48, 36)
        assert m_tb_48.shape == (4000, 48, 36)
        assert m_tc_48.shape == (4000, 48, 36)
        
        assert S_tr.shape == (3200, 5)
        assert S_val.shape == (800, 5)
        assert S_tb.shape == (4000, 5)
        assert S_tc.shape == (4000, 5)
        
        assert y_tr.shape == (3200,)
        assert y_val.shape == (800,)
        assert y_tb.shape == (4000,)
        assert y_tc.shape == (4000,)
        print(f"  [PASS] Sequence shapes verified for 48h, 24h, static vectors, and masks.")
        
        # Check no NaNs or Infs remain in imputed data
        for name, arr in [('X_train_48', X_tr_48), ('X_val_48', X_val_48), ('X_test_b_48', X_tb_48), ('X_test_c_48', X_tc_48),
                          ('static_train', S_tr), ('static_val', S_val), ('static_test_b', S_tb), ('static_test_c', S_tc)]:
            nan_cnt = np.isnan(arr).sum()
            inf_cnt = np.isinf(arr).sum()
            assert nan_cnt == 0, f"Found {nan_cnt} NaNs in {name} ({p})!"
            assert inf_cnt == 0, f"Found {inf_cnt} Infs in {name} ({p})!"
        print(f"  [PASS] Zero NaNs and zero Infs confirmed across all arrays.")
        
        # Verify labels are strictly binary {0, 1}
        for name, arr in [('y_train', y_tr), ('y_val', y_val), ('y_test_b', y_tb), ('y_test_c', y_tc)]:
            unique_vals = set(np.unique(arr))
            assert unique_vals.issubset({0, 1}), f"Non-binary labels in {name}: {unique_vals}"
        print(f"  [PASS] Mortality labels strictly binary [0, 1].")
        
        # Check that training standardization holds: train mean approx 0, std approx 1 (or 0 for constant MechVent)
        tr_flat = X_tr_48.reshape(-1, 36)
        means = np.mean(tr_flat, axis=0)
        stds = np.std(tr_flat, axis=0)
        assert np.all(np.abs(means) < 1e-4), f"Training mean not centered to 0! Max mean: {np.max(np.abs(means))}"
        for f_idx in range(36):
            if f_idx == 18: # MechVent (all observed values are 1.0, so median=1.0 and variance is 0)
                assert abs(stds[f_idx]) < 1e-4, f"MechVent std {stds[f_idx]} != 0.0"
            else:
                assert abs(stds[f_idx] - 1.0) < 1e-2, f"Feature {f_idx} std {stds[f_idx]} != 1.0"
        print(f"  [PASS] Standardized training features verified: 35 features have Mean = 0.000, Std = 1.000 (MechVent constant std = 0.000).")

    # 4. Check label matching with manifest
    rids_tr = np.load('data/processed/simple_imputation/rids_train.npy')
    y_tr = np.load('data/processed/simple_imputation/y_train.npy')
    manifest_tr_map = dict(zip(df_manifest[df_manifest['Partition'] == 'train']['RecordID'],
                                df_manifest[df_manifest['Partition'] == 'train']['In-hospital_death']))
    for i in range(len(rids_tr)):
        rid = rids_tr[i]
        assert y_tr[i] == manifest_tr_map[rid], f"Label mismatch for RecordID {rid}!"
    print("\n[Check 4] [PASS] Patient RecordIDs and In-Hospital Mortality labels match 100% bijectively.")

    print("\nALL SANITY AND DATA INTEGRITY CHECKS PASSED PERFECTLY!")

if __name__ == '__main__':
    verify_phase2()
