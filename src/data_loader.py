import os
import glob
import tarfile
import numpy as np
import pandas as pd

# 5 Static admission descriptors (Weight at admission is preserved as a static baseline)
STATIC_PARAMS = ['Age', 'Gender', 'Height', 'ICUType', 'Weight_admission']

# 37 Dynamic physiological time-series variables (including dynamic Weight)
DYNAMIC_PARAMS = [
    'ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin', 'Cholesterol', 
    'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 
    'HR', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg', 'NIDiasABP', 'NIMAP', 
    'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets', 'RespRate', 'SaO2', 
    'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC', 'Weight', 'pH'
]

# Ensure sorted order for reproducibility
DYNAMIC_PARAMS = sorted(DYNAMIC_PARAMS)

# 36 Dynamic parameters excluding dynamic Weight (for legacy/comparison support)
DYNAMIC_PARAMS_36 = sorted([p for p in DYNAMIC_PARAMS if p != 'Weight'])

def parse_time_to_minutes(time_str):
    """Convert HH:MM elapsed time string to integer minutes."""
    parts = time_str.split(':')
    return int(parts[0]) * 60 + int(parts[1])

def load_outcomes(dataset_dir='Dataset'):
    """Load and validate outcome labels for Set A, Set B, and Set C."""
    outcomes = {}
    for s in ['a', 'b', 'c']:
        p = os.path.join(dataset_dir, f'Outcomes-{s}.txt')
        df = pd.read_csv(p)
        df['Partition'] = f'Set {s.upper()}'
        outcomes[s] = df
    return outcomes

def load_patient_raw_records(dataset_dir='Dataset'):
    """
    Load raw records for all 12,000 patients across Set A, Set B, and Set C.
    Handles Set C directly from Dataset/set-c.tar.gz without modifying original files.
    Filters out corrupted records (empty parameter names) and logs counts.
    
    Returns:
        patient_data: dict of {record_id: {'static': dict, 'dynamic': list of (minutes, param, value)}}
        load_stats: dict of parsing statistics and dropped records log
    """
    patient_data = {}
    load_stats = {
        'total_files_parsed': 0,
        'records_per_partition': {'Set A': 0, 'Set B': 0, 'Set C': 0},
        'dropped_blank_param_lines': 0,
        'dropped_invalid_format_lines': 0,
        'total_observations_loaded': 0
    }
    
    # 1. Load Set A
    files_a = sorted(glob.glob(os.path.join(dataset_dir, 'set-a', '*.txt')))
    print(f"Loading {len(files_a)} patient files from Set A...")
    for fpath in files_a:
        rec_id = int(os.path.basename(fpath).replace('.txt', ''))
        with open(fpath, 'r') as fp:
            lines = fp.readlines()
        p_record = _parse_patient_lines(rec_id, lines, load_stats)
        p_record['partition'] = 'Set A'
        patient_data[rec_id] = p_record
        load_stats['records_per_partition']['Set A'] += 1
        load_stats['total_files_parsed'] += 1

    # 2. Load Set B
    files_b = sorted(glob.glob(os.path.join(dataset_dir, 'set-b', '*.txt')))
    print(f"Loading {len(files_b)} patient files from Set B...")
    for fpath in files_b:
        rec_id = int(os.path.basename(fpath).replace('.txt', ''))
        with open(fpath, 'r') as fp:
            lines = fp.readlines()
        p_record = _parse_patient_lines(rec_id, lines, load_stats)
        p_record['partition'] = 'Set B'
        patient_data[rec_id] = p_record
        load_stats['records_per_partition']['Set B'] += 1
        load_stats['total_files_parsed'] += 1

    # 3. Load Set C from archive
    tar_path = os.path.join(dataset_dir, 'set-c.tar.gz')
    print(f"Loading 4,000 patient files from Set C archive ({tar_path})...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.endswith('.txt'):
                rec_id = int(os.path.basename(m.name).replace('.txt', ''))
                f = tar.extractfile(m)
                lines = [line.decode('utf-8', errors='ignore') for line in f.readlines()]
                p_record = _parse_patient_lines(rec_id, lines, load_stats)
                p_record['partition'] = 'Set C'
                patient_data[rec_id] = p_record
                load_stats['records_per_partition']['Set C'] += 1
                load_stats['total_files_parsed'] += 1

    print(f"Successfully loaded {len(patient_data)} patient records.")
    print(f"Dropped blank parameter rows: {load_stats['dropped_blank_param_lines']}")
    print(f"Total valid observations loaded: {load_stats['total_observations_loaded']}")
    
    return patient_data, load_stats

def _parse_patient_lines(rec_id, lines, load_stats):
    """Helper to parse a single patient's file lines."""
    static_dict = {'RecordID': rec_id}
    dynamic_obs = []
    
    for line in lines[1:]: # skip header
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) != 3:
            load_stats['dropped_invalid_format_lines'] += 1
            continue
            
        t_str, param, val_str = parts[0].strip(), parts[1].strip(), parts[2].strip()
        
        # Check blank parameter
        if not param:
            load_stats['dropped_blank_param_lines'] += 1
            continue
            
        try:
            val = float(val_str)
        except ValueError:
            val = np.nan
            
        minutes = parse_time_to_minutes(t_str)
        
        # 1. Static descriptors handling
        if param in ['Age', 'Gender', 'Height', 'ICUType']:
            if minutes == 0 or param not in static_dict:
                static_dict[param] = val
        elif param == 'Weight':
            # Admission weight: prioritize 00:00 measurement or first available weight
            if 'Weight_admission' not in static_dict or (minutes == 0 and val > 0):
                static_dict['Weight_admission'] = val
            elif minutes == 0 and static_dict['Weight_admission'] == -1 and val > 0:
                static_dict['Weight_admission'] = val
                
        # 2. Dynamic physiological observations handling
        if param in DYNAMIC_PARAMS:
            dynamic_obs.append((minutes, param, val))
            load_stats['total_observations_loaded'] += 1
            
    # Sentinel conversion for static descriptors: -1 -> NaN
    for p in STATIC_PARAMS:
        if p not in static_dict:
            static_dict[p] = np.nan
        elif static_dict[p] == -1:
            static_dict[p] = np.nan
            
    return {
        'record_id': rec_id,
        'static': static_dict,
        'dynamic': dynamic_obs
    }
