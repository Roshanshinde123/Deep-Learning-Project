import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def parse_time_to_minutes(time_str):
    parts = time_str.split(':')
    return int(parts[0]) * 60 + int(parts[1])

def generate_eda_figures():
    print("=== GENERATING EDA VISUALIZATIONS & DETAILED STATS ===")
    os.makedirs('reports/figures', exist_ok=True)
    
    # 1. Outcomes Analysis
    df_a = pd.read_csv('Dataset/Outcomes-a.txt')
    df_b = pd.read_csv('Dataset/Outcomes-b.txt')
    df_c = pd.read_csv('Dataset/Outcomes-c.txt')
    
    df_a['Set'] = 'Set A (Train)'
    df_b['Set'] = 'Set B (Test/Ext)'
    df_c['Set'] = 'Set C (Held-out)'
    
    outcomes_combined = pd.concat([df_a, df_b, df_c], ignore_index=True)
    
    # FIGURE 1: Mortality Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Counts by Set
    sns.countplot(data=outcomes_combined, x='Set', hue='In-hospital_death', palette=['#2b5c8f', '#d95f02'], ax=axes[0])
    axes[0].set_title('In-Hospital Mortality Count by PhysioNet Set', fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel('Dataset Split', fontsize=11)
    axes[0].set_ylabel('Patient Count', fontsize=11)
    axes[0].legend(['Survivor (0)', 'Non-Survivor (1)'], title='Outcome', frameon=True)
    for p in axes[0].patches:
        h = p.get_height()
        if h > 0:
            axes[0].annotate(f'{int(h)}', (p.get_x() + p.get_width() / 2., h + 30),
                             ha='center', va='bottom', fontsize=9)
    
    # Right: Mortality Percentage
    mort_rates = outcomes_combined.groupby('Set')['In-hospital_death'].mean() * 100
    bars = axes[1].bar(mort_rates.index, mort_rates.values, color=['#1b9e77', '#d95f02', '#7570b3'], width=0.5)
    axes[1].set_title('In-Hospital Mortality Percentage (%)', fontsize=13, fontweight='bold', pad=12)
    axes[1].set_ylabel('Mortality Rate (%)', fontsize=11)
    axes[1].set_ylim(0, 20)
    for bar in bars:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.4, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('reports/figures/01_mortality_distribution.png', dpi=300)
    plt.close()
    print("Saved reports/figures/01_mortality_distribution.png")

    # Load summary inspection JSON for coverage
    with open('reports/dataset_inspection_summary.json', 'r') as f:
        insp = json.load(f)
    
    cov = insp['param_patient_coverage_pct']
    # Exclude RecordID
    cov_filtered = {k: v for k, v in cov.items() if k != 'RecordID'}
    cov_df = pd.DataFrame(list(cov_filtered.items()), columns=['Parameter', 'Coverage_Pct'])
    cov_df = cov_df.sort_values('Coverage_Pct', ascending=True)
    
    # FIGURE 2: Patient-Level Variable Coverage (% of Patients with >=1 measurement)
    plt.figure(figsize=(10, 11))
    colors = ['#d95f02' if c < 50 else '#2b5c8f' if c < 90 else '#1b9e77' for c in cov_df['Coverage_Pct']]
    bars = plt.barh(cov_df['Parameter'], cov_df['Coverage_Pct'], color=colors, height=0.65)
    plt.axvline(50, color='gray', linestyle='--', alpha=0.7, label='50% Coverage')
    plt.axvline(90, color='green', linestyle=':', alpha=0.7, label='90% Coverage')
    plt.title('Clinical Parameter Coverage Across 8,000 ICU Patients\n(% of Patients with at least 1 measurement in 48h)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Patient Coverage (%)', fontsize=11)
    plt.xlim(0, 105)
    for bar in bars:
        val = bar.get_width()
        plt.text(val + 1, bar.get_y() + bar.get_height()/2.0, f'{val:.1f}%', va='center', fontsize=8)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('reports/figures/02_variable_coverage.png', dpi=300)
    plt.close()
    print("Saved reports/figures/02_variable_coverage.png")

    # 3. Patient Hourly Vital Sign Extraction for Trajectories and Stats
    # Sample 1,000 Set A patients (500 survivors, 500 non-survivors if possible, or all 554 non-survivors + 1000 survivors)
    surv_ids = set(df_a[df_a['In-hospital_death'] == 0]['RecordID'].sample(800, random_state=42))
    death_ids = set(df_a[df_a['In-hospital_death'] == 1]['RecordID'])
    selected_ids = surv_ids.union(death_ids)
    
    records = []
    patient_obs_counts = []
    
    key_vitals = ['HR', 'GCS', 'Temp', 'RespRate', 'Urine', 'BUN', 'Creatinine', 'Glucose', 'WBC', 'NIMAP', 'MAP', 'DiasABP', 'SysABP', 'HCO3', 'K', 'Na', 'HCT', 'Platelets']
    
    files_a = glob.glob('Dataset/set-a/*.txt')
    patient_outcomes = dict(zip(df_a['RecordID'], df_a['In-hospital_death']))
    
    hourly_records = []
    patient_aggregates = []

    print(f"Sampling and extracting hourly features from Set A for figures...")
    for fpath in files_a:
        rec_id = int(os.path.basename(fpath).replace('.txt', ''))
        with open(fpath, 'r') as f:
            lines = [l.strip().split(',') for l in f.readlines() if l.strip()]
        
        obs_count = len(lines) - 1
        patient_obs_counts.append(obs_count)
        
        if rec_id in selected_ids:
            ihd = patient_outcomes.get(rec_id, 0)
            patient_vals = {v: [] for v in key_vitals}
            
            for row in lines[1:]:
                if len(row) != 3:
                    continue
                t, p, val_str = row[0], row[1], row[2]
                try:
                    val = float(val_str)
                except ValueError:
                    continue
                
                minutes = parse_time_to_minutes(t)
                hour = min(minutes // 60, 47)
                
                if p in key_vitals:
                    patient_vals[p].append(val)
                    if p in ['HR', 'Temp', 'GCS', 'NIMAP', 'RespRate']:
                        hourly_records.append({
                            'RecordID': rec_id,
                            'Hour': hour,
                            'Parameter': p,
                            'Value': val,
                            'Outcome': 'Non-Survivor' if ihd == 1 else 'Survivor'
                        })
            
            agg = {'RecordID': rec_id, 'Outcome': ihd}
            for v in key_vitals:
                arr = patient_vals[v]
                agg[f'{v}_mean'] = np.nanmean(arr) if arr else np.nan
                agg[f'{v}_min'] = np.nanmin(arr) if arr else np.nan
                agg[f'{v}_max'] = np.nanmax(arr) if arr else np.nan
                agg[f'{v}_last'] = arr[-1] if arr else np.nan
            patient_aggregates.append(agg)
            
    df_hourly = pd.DataFrame(hourly_records)
    df_agg = pd.DataFrame(patient_aggregates)

    # FIGURE 3: Observation Count Distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(patient_obs_counts, bins=50, kde=True, color='#2b5c8f')
    plt.axvline(np.median(patient_obs_counts), color='red', linestyle='--', label=f'Median: {int(np.median(patient_obs_counts))} obs')
    plt.title('Distribution of Total Clinical Observations Per Patient (First 48 Hours)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Total Observations Recorded per Patient', fontsize=11)
    plt.ylabel('Patient Count', fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig('reports/figures/03_observation_frequency.png', dpi=300)
    plt.close()
    print("Saved reports/figures/03_observation_frequency.png")

    # FIGURE 4: Key Vitals Distribution by Mortality Status (Boxplots)
    features_to_plot = ['HR_mean', 'GCS_min', 'Temp_mean', 'BUN_max', 'Creatinine_max', 'WBC_max']
    titles = ['Heart Rate Mean (bpm)', 'Min GCS Score', 'Temp Mean (°C)', 'Max BUN (mg/dL)', 'Max Creatinine (mg/dL)', 'Max WBC (x10^3/µL)']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    
    for i, feat in enumerate(features_to_plot):
        data_clean = df_agg[['Outcome', feat]].dropna()
        # filter extreme outliers for visual clarity
        q99 = data_clean[feat].quantile(0.99)
        q01 = data_clean[feat].quantile(0.01)
        data_clean = data_clean[(data_clean[feat] >= q01) & (data_clean[feat] <= q99)]
        
        sns.boxplot(data=data_clean, x='Outcome', y=feat, palette=['#2b5c8f', '#d95f02'], ax=axes[i], width=0.45)
        axes[i].set_xticklabels(['Survivor', 'Non-Survivor'], fontsize=10, fontweight='bold')
        axes[i].set_title(titles[i], fontsize=12, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
    
    plt.suptitle('Comparison of Key Physiological Variables: Survivors vs Non-Survivors', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('reports/figures/04_vitals_by_outcome.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved reports/figures/04_vitals_by_outcome.png")

    # FIGURE 5: Hourly Physiological Trajectories (48 Hours)
    vitals_traj = ['HR', 'Temp', 'GCS', 'NIMAP']
    v_titles = ['Heart Rate (bpm)', 'Temperature (°C)', 'Glasgow Coma Scale (GCS)', 'Non-Invasive Mean ABP (mmHg)']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, v in enumerate(vitals_traj):
        sub = df_hourly[df_hourly['Parameter'] == v].copy()
        # Filter physiological bounds for plot
        if v == 'HR':
            sub = sub[(sub['Value'] > 30) & (sub['Value'] < 200)]
        elif v == 'Temp':
            sub = sub[(sub['Value'] > 30) & (sub['Value'] < 42)]
        elif v == 'GCS':
            sub = sub[(sub['Value'] >= 3) & (sub['Value'] <= 15)]
        elif v == 'NIMAP':
            sub = sub[(sub['Value'] > 20) & (sub['Value'] < 180)]
            
        sns.lineplot(data=sub, x='Hour', y='Value', hue='Outcome', 
                     palette={'Survivor': '#2b5c8f', 'Non-Survivor': '#d95f02'}, 
                     ax=axes[i], errorbar=('ci', 95))
        axes[i].set_title(f'48-Hour Trajectory: {v_titles[i]}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('ICU Admission Hour', fontsize=10)
        axes[i].set_ylabel(v, fontsize=10)
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].legend(title='Outcome', frameon=True)
    
    plt.suptitle('Temporal Trajectories Over the First 48 Hours in ICU (95% CI)', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('reports/figures/05_temporal_trajectories_48h.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved reports/figures/05_temporal_trajectories_48h.png")

    # FIGURE 6: Correlation Heatmap
    corr_cols = ['Outcome', 'HR_mean', 'GCS_min', 'Temp_mean', 'BUN_max', 'Creatinine_max', 'WBC_max', 'NIMAP_mean', 'HCO3_min', 'K_max', 'Na_mean', 'Platelets_min']
    df_corr = df_agg[corr_cols].dropna()
    corr_matrix = df_corr.corr(method='spearman')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-0.5, vmax=0.5, linewidths=0.5)
    plt.title('Spearman Correlation Matrix of Physiological Features & Mortality', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('reports/figures/06_correlation_heatmap.png', dpi=300)
    plt.close()
    print("Saved reports/figures/06_correlation_heatmap.png")

    print("\nAll 6 figures generated successfully!")

if __name__ == '__main__':
    generate_eda_figures()
