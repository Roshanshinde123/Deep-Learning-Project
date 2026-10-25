import os
import glob
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

def parse_time_to_minutes(time_str):
    parts = time_str.split(':')
    return int(parts[0]) * 60 + int(parts[1])

def inspect_dataset():
    print("=== STARTING FULL DATASET INSPECTION ===")
    
    # 1. Outcomes Inspection
    outcomes = {}
    for s in ['a', 'b', 'c']:
        p = f'Dataset/Outcomes-{s}.txt'
        outcomes[s] = pd.read_csv(p)
        print(f"Loaded Outcomes-{s}: {len(outcomes[s])} records")

    # 2. Inspect Set A and Set B files
    sets = {'set-a': 'Dataset/set-a', 'set-b': 'Dataset/set-b'}
    
    # Static parameters
    static_params = ['RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight']
    
    dataset_summary = {
        'total_records_set_a': len(outcomes['a']),
        'total_records_set_b': len(outcomes['b']),
        'total_records_set_c': len(outcomes['c']),
        'params_found': Counter(),
        'static_data': {'set-a': [], 'set-b': []},
        'time_distribution': {'max_minutes': [], 'total_obs_per_patient': []},
        'duplicate_measurements': 0,
        'duplicate_measurements_same_val': 0,
        'duplicate_measurements_diff_val': 0,
        'obs_beyond_48h': 0,
        'per_param_stats': defaultdict(list),
        'per_param_patient_count': defaultdict(int),
        'param_per_set_count': {'set-a': Counter(), 'set-b': Counter()}
    }
    
    all_rows = []
    
    for set_name, dir_path in sets.items():
        files = glob.glob(os.path.join(dir_path, '*.txt'))
        print(f"Parsing {len(files)} files in {set_name}...")
        
        for idx, fpath in enumerate(files):
            rec_id = int(os.path.basename(fpath).replace('.txt', ''))
            with open(fpath, 'r') as f:
                lines = f.readlines()
            
            header = lines[0].strip().split(',')
            
            patient_static = {'RecordID': rec_id}
            seen_time_params = {}
            patient_params = set()
            obs_count = 0
            
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) != 3:
                    continue
                t_str, param, val_str = parts[0], parts[1], parts[2]
                
                try:
                    val = float(val_str)
                except ValueError:
                    val = np.nan
                
                minutes = parse_time_to_minutes(t_str)
                
                if minutes > 48 * 60:
                    dataset_summary['obs_beyond_48h'] += 1
                
                if param in static_params and minutes == 0:
                    patient_static[param] = val
                
                patient_params.add(param)
                dataset_summary['params_found'][param] += 1
                dataset_summary['param_per_set_count'][set_name][param] += 1
                obs_count += 1
                
                # Check duplicates at exact same minute & param
                key = (minutes, param)
                if key in seen_time_params:
                    dataset_summary['duplicate_measurements'] += 1
                    if seen_time_params[key] == val:
                        dataset_summary['duplicate_measurements_same_val'] += 1
                    else:
                        dataset_summary['duplicate_measurements_diff_val'] += 1
                else:
                    seen_time_params[key] = val
                
                # Keep sample values for numeric stats (sample up to 5000 per param)
                if len(dataset_summary['per_param_stats'][param]) < 10000:
                    dataset_summary['per_param_stats'][param].append(val)
            
            for p in patient_params:
                dataset_summary['per_param_patient_count'][p] += 1
            
            dataset_summary['static_data'][set_name].append(patient_static)
            dataset_summary['time_distribution']['total_obs_per_patient'].append(obs_count)
            if lines[1:]:
                last_time = parse_time_to_minutes(lines[-1].strip().split(',')[0])
                dataset_summary['time_distribution']['max_minutes'].append(last_time)
            
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(files)} in {set_name}")

    print("\nParsing complete! Compiling statistics...")
    
    # Static dataframes
    df_static_a = pd.DataFrame(dataset_summary['static_data']['set-a'])
    df_static_b = pd.DataFrame(dataset_summary['static_data']['set-b'])
    
    print("\n--- STATIC VARIABLES SUMMARY (Set A) ---")
    print(df_static_a.describe().to_string())
    print("\n--- STATIC VARIABLES SUMMARY (Set B) ---")
    print(df_static_b.describe().to_string())
    
    # Save processed inspect metrics
    static_summary = {
        'set_a_static': df_static_a.describe().to_dict(),
        'set_b_static': df_static_b.describe().to_dict(),
        'unique_params': sorted(list(dataset_summary['params_found'].keys())),
        'total_observations': sum(dataset_summary['params_found'].values()),
        'param_counts': dict(dataset_summary['params_found']),
        'param_patient_coverage_pct': {p: count / 8000.0 * 100 for p, count in dataset_summary['per_param_patient_count'].items()},
        'duplicate_measurements': dataset_summary['duplicate_measurements'],
        'duplicate_measurements_same_val': dataset_summary['duplicate_measurements_same_val'],
        'duplicate_measurements_diff_val': dataset_summary['duplicate_measurements_diff_val'],
        'obs_beyond_48h': dataset_summary['obs_beyond_48h'],
        'max_obs_minutes': max(dataset_summary['time_distribution']['max_minutes']) if dataset_summary['time_distribution']['max_minutes'] else 0
    }
    
    # Check physiological bounds and potential invalid values
    param_ranges = {}
    for p, vals in dataset_summary['per_param_stats'].items():
        arr = np.array(vals)
        arr_clean = arr[~np.isnan(arr)]
        if len(arr_clean) > 0:
            param_ranges[p] = {
                'min': float(np.min(arr_clean)),
                'q01': float(np.percentile(arr_clean, 1)),
                'median': float(np.median(arr_clean)),
                'q99': float(np.percentile(arr_clean, 99)),
                'max': float(np.max(arr_clean)),
                'negative_count': int(np.sum(arr_clean < 0)),
                'zero_count': int(np.sum(arr_clean == 0))
            }
    
    static_summary['param_ranges'] = param_ranges
    
    with open('reports/dataset_inspection_summary.json', 'w') as f:
        json.dump(static_summary, f, indent=2)
    print("\nSaved comprehensive inspection to reports/dataset_inspection_summary.json")

if __name__ == '__main__':
    inspect_dataset()
