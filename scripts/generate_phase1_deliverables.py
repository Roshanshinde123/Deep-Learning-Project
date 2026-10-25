import os
import glob
import json
import tarfile
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def parse_time_to_minutes(time_str):
    parts = time_str.split(':')
    return int(parts[0]) * 60 + int(parts[1])

# Biological plausibility bounds
BIOLOGICAL_BOUNDS = {
    'HR': (20, 250),           # bpm
    'Temp': (25.0, 43.0),      # Celsius
    'SysABP': (30, 300),       # mmHg
    'DiasABP': (10, 200),      # mmHg
    'MAP': (20, 250),          # mmHg
    'NISysABP': (30, 300),     # mmHg
    'NIDiasABP': (10, 200),    # mmHg
    'NIMAP': (20, 250),        # mmHg
    'RespRate': (4, 80),       # breaths/min
    'GCS': (3, 15),            # score
    'pH': (6.7, 7.8),          # -
    'PaO2': (15, 600),         # mmHg
    'PaCO2': (10, 150),        # mmHg
    'SaO2': (50, 100),         # %
    'FiO2': (0.21, 1.0),       # ratio
    'Urine': (0, 5000),        # mL (single output)
    'BUN': (1, 250),           # mg/dL
    'Creatinine': (0.1, 30),   # mg/dL
    'Glucose': (10, 1500),     # mg/dL
    'K': (1.0, 12.0),          # mEq/L
    'Na': (90, 185),           # mEq/L
    'WBC': (0.1, 200),         # x10^3/uL
    'Platelets': (2, 2000),    # x10^3/uL
    'HCT': (5, 75),            # %
    'Height': (50, 250),       # cm
    'Weight': (20, 450),       # kg
}

def run_phase1_analysis():
    print("=== STARTING FULL PHASE 1 COMPREHENSIVE AUDIT (12,000 PATIENTS) ===")
    os.makedirs('reports', exist_ok=True)
    
    # 1. Outcomes Inspection
    outcomes = {}
    for s in ['a', 'b', 'c']:
        df = pd.read_csv(f'Dataset/Outcomes-{s}.txt')
        outcomes[s] = df
        print(f"Loaded Outcomes-{s}: {len(df)} records, columns: {df.columns.tolist()}")

    df_a, df_b, df_c = outcomes['a'], outcomes['b'], outcomes['c']
    df_a['Partition'] = 'Set A'
    df_b['Partition'] = 'Set B'
    df_c['Partition'] = 'Set C'
    combined_outcomes = pd.concat([df_a, df_b, df_c], ignore_index=True)
    
    # 2. Setup stats collectors
    all_static_params = ['RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight']
    
    partition_stats = {
        'Set A': {'records': 0, 'observations': 0, 'duplicates': 0, 'beyond_48h': 0, 'files': 0},
        'Set B': {'records': 0, 'observations': 0, 'duplicates': 0, 'beyond_48h': 0, 'files': 0},
        'Set C': {'records': 0, 'observations': 0, 'duplicates': 0, 'beyond_48h': 0, 'files': 0}
    }
    
    # Variable tracking: {param: {count: N, patient_ids: set(), values: []}}
    param_data = defaultdict(lambda: {'count': 0, 'patient_ids': set(), 'values': [], 'invalid_count': 0})
    param_per_partition = {
        'Set A': defaultdict(lambda: {'count': 0, 'patients': set()}),
        'Set B': defaultdict(lambda: {'count': 0, 'patients': set()}),
        'Set C': defaultdict(lambda: {'count': 0, 'patients': set()})
    }
    
    patient_obs_counts = []
    hourly_obs_counts = [0] * 48
    all_seen_patient_ids = set()
    
    # A. Parse Set A
    print("\nParsing Set A (4,000 files in Dataset/set-a)...")
    files_a = glob.glob('Dataset/set-a/*.txt')
    partition_stats['Set A']['files'] = len(files_a)
    for idx, fpath in enumerate(files_a):
        rec_id = int(os.path.basename(fpath).replace('.txt', ''))
        all_seen_patient_ids.add(rec_id)
        partition_stats['Set A']['records'] += 1
        with open(fpath, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        _process_patient_lines(rec_id, lines, 'Set A', partition_stats, param_data, param_per_partition, patient_obs_counts, hourly_obs_counts)
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/4000 in Set A")

    # B. Parse Set B
    print("\nParsing Set B (4,000 files in Dataset/set-b)...")
    files_b = glob.glob('Dataset/set-b/*.txt')
    partition_stats['Set B']['files'] = len(files_b)
    for idx, fpath in enumerate(files_b):
        rec_id = int(os.path.basename(fpath).replace('.txt', ''))
        all_seen_patient_ids.add(rec_id)
        partition_stats['Set B']['records'] += 1
        with open(fpath, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        _process_patient_lines(rec_id, lines, 'Set B', partition_stats, param_data, param_per_partition, patient_obs_counts, hourly_obs_counts)
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/4000 in Set B")

    # C. Parse Set C directly from set-c.tar.gz without modifying original files
    print("\nParsing Set C (4,000 files inside Dataset/set-c.tar.gz)...")
    with tarfile.open('Dataset/set-c.tar.gz', 'r:gz') as tar:
        txt_members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith('.txt')]
        partition_stats['Set C']['files'] = len(txt_members)
        for idx, m in enumerate(txt_members):
            rec_id = int(os.path.basename(m.name).replace('.txt', ''))
            all_seen_patient_ids.add(rec_id)
            partition_stats['Set C']['records'] += 1
            f = tar.extractfile(m)
            lines = [l.decode('utf-8', errors='ignore').strip() for l in f.readlines() if l.strip()]
            _process_patient_lines(rec_id, lines, 'Set C', partition_stats, param_data, param_per_partition, patient_obs_counts, hourly_obs_counts)
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/4000 in Set C")

    total_patients = len(all_seen_patient_ids)
    print(f"\nTotal unique patients parsed across Set A, B, C: {total_patients}")

    # 3. Create dataset_summary.json
    print("\nBuilding dataset_summary.json...")
    summary_json = {
        'dataset_title': 'PhysioNet / Computing in Cardiology Challenge 2012',
        'clinical_objective': 'In-Hospital Mortality Risk Prediction from First 48 Hours',
        'partitions': {
            'set_a': {
                'directory': 'Dataset/set-a',
                'outcomes_file': 'Dataset/Outcomes-a.txt',
                'patient_count': partition_stats['Set A']['records'],
                'observation_count': partition_stats['Set A']['observations'],
                'duplicate_timestamp_collisions': partition_stats['Set A']['duplicates'],
                'survivors': int((df_a['In-hospital_death'] == 0).sum()),
                'deaths': int((df_a['In-hospital_death'] == 1).sum()),
                'mortality_rate_pct': float(df_a['In-hospital_death'].mean() * 100),
                'role': 'Official Development / Training Partition'
            },
            'set_b': {
                'directory': 'Dataset/set-b',
                'outcomes_file': 'Dataset/Outcomes-b.txt',
                'patient_count': partition_stats['Set B']['records'],
                'observation_count': partition_stats['Set B']['observations'],
                'duplicate_timestamp_collisions': partition_stats['Set B']['duplicates'],
                'survivors': int((df_b['In-hospital_death'] == 0).sum()),
                'deaths': int((df_b['In-hospital_death'] == 1).sum()),
                'mortality_rate_pct': float(df_b['In-hospital_death'].mean() * 100),
                'role': 'Official Competition Test / Leaderboard Evaluation Partition'
            },
            'set_c': {
                'archive': 'Dataset/set-c.tar.gz',
                'outcomes_file': 'Dataset/Outcomes-c.txt',
                'patient_count': partition_stats['Set C']['records'],
                'observation_count': partition_stats['Set C']['observations'],
                'duplicate_timestamp_collisions': partition_stats['Set C']['duplicates'],
                'survivors': int((df_c['In-hospital_death'] == 0).sum()),
                'deaths': int((df_c['In-hospital_death'] == 1).sum()),
                'mortality_rate_pct': float(df_c['In-hospital_death'].mean() * 100),
                'role': 'Official Final Evaluation / Held-Out Challenge Partition'
            },
            'combined_all_partitions': {
                'total_patient_records': total_patients,
                'total_unique_patient_ids': len(all_seen_patient_ids),
                'total_observations': sum(p['observations'] for p in partition_stats.values()),
                'total_duplicate_collisions': sum(p['duplicates'] for p in partition_stats.values()),
                'total_survivors': int((combined_outcomes['In-hospital_death'] == 0).sum()),
                'total_deaths': int((combined_outcomes['In-hospital_death'] == 1).sum()),
                'overall_mortality_rate_pct': float(combined_outcomes['In-hospital_death'].mean() * 100),
                'imbalance_ratio': f"1 : {(combined_outcomes['In-hospital_death'] == 0).sum() / (combined_outcomes['In-hospital_death'] == 1).sum():.2f}"
            }
        },
        'phase1_directory_explanation': {
            'path': 'Dataset/phase1',
            'contents': 'Archival copy of early competition Phase 1 data (set-a, set-b archives and folders)',
            'key_discrepancy': 'Phase 1 files lack the ICUType variable (added in Phase 2 final challenge release). Root set-a and set-b are the updated final versions.'
        },
        'temporal_characteristics': {
            'timestamp_format': 'HH:MM (elapsed hours and minutes since ICU admission)',
            'time_range_minutes': [0, 2880],
            'time_range_hours': [0, 48],
            'observations_beyond_48h': sum(p['beyond_48h'] for p in partition_stats.values())
        }
    }
    
    with open('reports/dataset_summary.json', 'w') as f:
        json.dump(summary_json, f, indent=2)
    print("Saved reports/dataset_summary.json")

    # 4. Create variable_summary.csv and missingness_report.csv
    print("\nBuilding variable_summary.csv & missingness_report.csv...")
    var_rows = []
    miss_rows = []
    
    for param in sorted(param_data.keys()):
        if param == 'RecordID':
            continue
        pinfo = param_data[param]
        vals = np.array(pinfo['values'])
        valid_vals = vals[~np.isnan(vals)]
        
        # Clinical categorization
        cat = _get_clinical_category(param)
        
        # Missingness across total 12,000 patients
        pat_cov = len(pinfo['patient_ids'])
        pat_cov_pct = (pat_cov / total_patients) * 100.0
        pat_miss_pct = 100.0 - pat_cov_pct
        
        # Partition missingness
        miss_a = 100.0 - (len(param_per_partition['Set A'][param]['patients']) / 4000.0 * 100.0)
        miss_b = 100.0 - (len(param_per_partition['Set B'][param]['patients']) / 4000.0 * 100.0)
        miss_c = 100.0 - (len(param_per_partition['Set C'][param]['patients']) / 4000.0 * 100.0)
        
        # Stats
        if len(valid_vals) > 0:
            vmin = float(np.min(valid_vals))
            vq25 = float(np.percentile(valid_vals, 25))
            vmed = float(np.median(valid_vals))
            vq75 = float(np.percentile(valid_vals, 75))
            vmax = float(np.max(valid_vals))
            vmean = float(np.mean(valid_vals))
            vstd = float(np.std(valid_vals))
        else:
            vmin = vq25 = vmed = vq75 = vmax = vmean = vstd = np.nan
            
        b_range = BIOLOGICAL_BOUNDS.get(param, ('N/A', 'N/A'))
        b_str = f"[{b_range[0]}, {b_range[1]}]" if b_range[0] != 'N/A' else 'General'
        
        var_rows.append({
            'Parameter': param,
            'Category': cat,
            'Total_Observations': pinfo['count'],
            'Patients_With_Measurement': pat_cov,
            'Patient_Coverage_Pct': round(pat_cov_pct, 2),
            'Patient_Missing_Pct': round(pat_miss_pct, 2),
            'Min': round(vmin, 2) if not np.isnan(vmin) else 'NaN',
            'Q25': round(vq25, 2) if not np.isnan(vq25) else 'NaN',
            'Median': round(vmed, 2) if not np.isnan(vmed) else 'NaN',
            'Q75': round(vq75, 2) if not np.isnan(vq75) else 'NaN',
            'Max': round(vmax, 2) if not np.isnan(vmax) else 'NaN',
            'Mean': round(vmean, 2) if not np.isnan(vmean) else 'NaN',
            'Std': round(vstd, 2) if not np.isnan(vstd) else 'NaN',
            'Invalid_Outlier_Count': pinfo['invalid_count'],
            'Physiological_Range': b_str
        })
        
        miss_rows.append({
            'Parameter': param,
            'Category': cat,
            'Total_Obs': pinfo['count'],
            'Coverage_All_Pct': round(pat_cov_pct, 2),
            'Missing_All_Pct': round(pat_miss_pct, 2),
            'Missing_SetA_Pct': round(miss_a, 2),
            'Missing_SetB_Pct': round(miss_b, 2),
            'Missing_SetC_Pct': round(miss_c, 2),
            'Unrecorded_vs_Zero_Note': 'Measurement was unrecorded (sparse testing); must NOT be treated as zero.' if cat != 'Static Descriptor' else 'Demographic or admission variable.'
        })

    df_var_summary = pd.DataFrame(var_rows).sort_values('Total_Observations', ascending=False)
    df_var_summary.to_csv('reports/variable_summary.csv', index=False)
    print("Saved reports/variable_summary.csv")

    df_miss = pd.DataFrame(miss_rows).sort_values('Missing_All_Pct', ascending=True)
    df_miss.to_csv('reports/missingness_report.csv', index=False)
    print("Saved reports/missingness_report.csv")

    # 5. GENERATE REQUIRED VISUALIZATIONS
    print("\nGenerating requested visualization figures...")
    
    # FIGURE A: outcome_distribution.png
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(data=combined_outcomes, x='Partition', hue='In-hospital_death', palette=['#1f77b4', '#d62728'], ax=axes[0])
    axes[0].set_title('In-Hospital Mortality Count by PhysioNet 2012 Partition\n(12,000 Total Patients: Set A, Set B, Set C)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Patient Count')
    axes[0].set_xlabel('Dataset Partition')
    axes[0].legend(['Survivor (0)', 'Non-Survivor (1)'], title='Outcome')
    for p in axes[0].patches:
        h = p.get_height()
        if h > 0:
            axes[0].annotate(f'{int(h)}', (p.get_x() + p.get_width() / 2., h + 30), ha='center', va='bottom', fontsize=9)
            
    # Percentage plot
    mort_rates = combined_outcomes.groupby('Partition')['In-hospital_death'].mean() * 100
    bars = axes[1].bar(mort_rates.index, mort_rates.values, color=['#1f77b4', '#ff7f0e', '#2ca02c'], width=0.5)
    axes[1].set_title('In-Hospital Mortality Percentage (%)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Mortality Rate (%)')
    axes[1].set_ylim(0, 20)
    for bar in bars:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.4, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/outcome_distribution.png', dpi=300)
    plt.close()
    print("Saved reports/outcome_distribution.png")

    # FIGURE B: missingness_visualization.png
    plt.figure(figsize=(11, 12))
    df_miss_plot = df_miss.copy()
    colors = []
    for m in df_miss_plot['Missing_All_Pct']:
        if m < 15:
            colors.append('#2ca02c') # green
        elif m < 50:
            colors.append('#1f77b4') # blue
        elif m < 80:
            colors.append('#ff7f0e') # orange
        else:
            colors.append('#d62728') # red
            
    bars = plt.barh(df_miss_plot['Parameter'], df_miss_plot['Missing_All_Pct'], color=colors, height=0.65)
    plt.axvline(15, color='gray', linestyle='--', alpha=0.7, label='15% Missing (Routine Labs / Vitals)')
    plt.axvline(50, color='blue', linestyle=':', alpha=0.7, label='50% Missing (Moderate)')
    plt.axvline(80, color='red', linestyle='-.', alpha=0.7, label='80% Missing (Specialized/Infrequent)')
    plt.title('Patient-Level Missingness Profile Across All 12,000 ICU Patients\n(% of Patients with NO Measurement in First 48h)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Missingness Percentage (%) [Unrecorded in 48h]', fontsize=11)
    plt.xlim(0, 105)
    for bar in bars:
        val = bar.get_width()
        plt.text(val + 1, bar.get_y() + bar.get_height()/2.0, f'{val:.1f}%', va='center', fontsize=8)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig('reports/missingness_visualization.png', dpi=300)
    plt.close()
    print("Saved reports/missingness_visualization.png")

    # FIGURE C: measurement_distribution.png
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(patient_obs_counts, bins=50, kde=True, color='#1f77b4', ax=axes[0])
    axes[0].axvline(np.median(patient_obs_counts), color='red', linestyle='--', label=f'Median: {int(np.median(patient_obs_counts))} obs')
    axes[0].set_title('Distribution of Observation Counts per Patient (First 48 Hours)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Total Observations Recorded per Patient')
    axes[0].set_ylabel('Patient Count')
    axes[0].legend()

    # Right: hourly measurement volume
    hours = list(range(48))
    axes[1].bar(hours, hourly_obs_counts, color='#2b5c8f', width=0.8)
    axes[1].set_title('Total Measurements Recorded per Hour of ICU Admission', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('ICU Admission Hour (0 to 47)')
    axes[1].set_ylabel('Total Observations Across 12,000 Patients')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('reports/measurement_distribution.png', dpi=300)
    plt.close()
    print("Saved reports/measurement_distribution.png")

    print("\nPhase 1 deliverables generated successfully!")

def _process_patient_lines(rec_id, lines, partition_name, partition_stats, param_data, param_per_partition, patient_obs_counts, hourly_obs_counts):
    seen_time_param = set()
    patient_params = set()
    obs_count = 0
    
    for line in lines[1:]:
        parts = line.split(',')
        if len(parts) != 3:
            continue
        t_str, param, val_str = parts[0], parts[1], parts[2]
        try:
            val = float(val_str)
        except ValueError:
            val = np.nan
            
        minutes = parse_time_to_minutes(t_str)
        if minutes > 2880:
            partition_stats[partition_name]['beyond_48h'] += 1
            
        hour = min(minutes // 60, 47)
        hourly_obs_counts[hour] += 1
        obs_count += 1
        
        # Check duplicate
        key = (minutes, param)
        if key in seen_time_param:
            partition_stats[partition_name]['duplicates'] += 1
        else:
            seen_time_param.add(key)
            
        # Check biological validity
        is_invalid = False
        if param in BIOLOGICAL_BOUNDS and not np.isnan(val):
            low, high = BIOLOGICAL_BOUNDS[param]
            if val < low or val > high:
                is_invalid = True
        elif param in ['Gender', 'Height', 'Weight'] and val == -1:
            # -1 is missing sentinel
            is_invalid = True
            
        if is_invalid:
            param_data[param]['invalid_count'] += 1
            
        param_data[param]['count'] += 1
        param_data[param]['patient_ids'].add(rec_id)
        if len(param_data[param]['values']) < 15000:
            param_data[param]['values'].append(val)
            
        param_per_partition[partition_name][param]['count'] += 1
        param_per_partition[partition_name][param]['patients'].add(rec_id)
        
    partition_stats[partition_name]['observations'] += obs_count
    patient_obs_counts.append(obs_count)

def _get_clinical_category(param):
    if param in ['Age', 'Gender', 'Height', 'ICUType', 'Weight']:
        return 'Static Descriptor'
    elif param in ['HR', 'Temp', 'Urine', 'RespRate']:
        return 'Core Vital Signs'
    elif param in ['NISysABP', 'NIDiasABP', 'NIMAP']:
        return 'Non-Invasive Hemodynamics'
    elif param in ['SysABP', 'DiasABP', 'MAP']:
        return 'Invasive Arterial Hemodynamics'
    elif param == 'GCS':
        return 'Neurological Function'
    elif param in ['BUN', 'Creatinine', 'Glucose', 'HCO3', 'K', 'Na', 'Mg']:
        return 'Serum Chemistry & Renal'
    elif param in ['HCT', 'Platelets', 'WBC']:
        return 'Hematology'
    elif param in ['MechVent', 'FiO2', 'PaO2', 'PaCO2', 'pH', 'SaO2']:
        return 'Arterial Blood Gas & Ventilation'
    elif param in ['Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin']:
        return 'Hepatic Enzymes & Function'
    elif param in ['Lactate', 'Cholesterol', 'TroponinI', 'TroponinT']:
        return 'Metabolic & Cardiac Biomarkers'
    else:
        return 'Other Clinical'

if __name__ == '__main__':
    run_phase1_analysis()
