import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def generate_shap_analysis(model, X, feature_names, y_true=None, y_prob=None,
                           max_display=15, 
                           summary_plot_path='reports/xgboost_shap_summary.png',
                           importance_plot_path='reports/xgboost_feature_importance.png',
                           patient_plots_dir='reports/figures/'):
    """
    Performs comprehensive SHAP-based model explainability for the fitted XGBoost classifier.
    
    Parameters:
        model: fitted xgb.XGBClassifier
        X: np.ndarray of shape (N, D) - feature matrix (e.g. Validation or Test cohort)
        feature_names: list of D feature names
        y_true: optional binary ground truth labels
        y_prob: optional predicted probabilities
        max_display: int, number of top features to highlight
        
    Returns:
        df_importance: pd.DataFrame with features ranked by mean |SHAP| value
        shap_explanation: shap.Explanation object
    """
    os.makedirs(os.path.dirname(summary_plot_path), exist_ok=True)
    os.makedirs(os.path.dirname(importance_plot_path), exist_ok=True)
    os.makedirs(patient_plots_dir, exist_ok=True)
    
    print("\nComputing TreeExplainer SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_explanation = explainer(X)
    
    # Assign feature names to explanation object
    shap_explanation.feature_names = feature_names
    shap_values = shap_explanation.values # (N, D)
    
    # -------------------------------------------------------------
    # 1. GLOBAL FEATURE IMPORTANCE (MEAN ABSOLUTE SHAP VALUE)
    # -------------------------------------------------------------
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
    
    # Also include XGBoost's built-in feature importances (gain & weight)
    booster = model.get_booster()
    score_gain = booster.get_score(importance_type='gain')
    score_weight = booster.get_score(importance_type='weight')
    
    df_importance['XGB_Gain'] = [score_gain.get(f"f{i}", 0.0) for i in range(len(feature_names))]
    df_importance['XGB_Weight'] = [score_weight.get(f"f{i}", 0.0) for i in range(len(feature_names))]
    
    # -------------------------------------------------------------
    # 2. SHAP SUMMARY PLOT (BEESWARM)
    # -------------------------------------------------------------
    print(f"Generating SHAP summary plot: {summary_plot_path}...")
    fig = plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_explanation, max_display=max_display, show=False)
    plt.title(f'SHAP Summary: Top {max_display} ICU Mortality Predictors', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # -------------------------------------------------------------
    # 3. TOP 15 FEATURE IMPORTANCE BAR PLOT
    # -------------------------------------------------------------
    print(f"Generating Top 15 Feature Importance plot: {importance_plot_path}...")
    top15 = df_importance.head(max_display).sort_values('Mean_Abs_SHAP', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars = ax.barh(top15['Feature'], top15['Mean_Abs_SHAP'], color='#2b5c8f', edgecolor='black', alpha=0.85)
    ax.set_title(f'Top {max_display} Most Influential Predictors of ICU Mortality (|SHAP|)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Mean Absolute SHAP Value (Impact on Log-Odds of Mortality)', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6, axis='x')
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height()/2.0, f"{w:.3f}", va='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(importance_plot_path, dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # 4. INDIVIDUAL PATIENT EXPLANATIONS (HIGH RISK VS LOW RISK)
    # -------------------------------------------------------------
    if y_prob is not None:
        # Find highest risk predicted non-survivor (True Positive candidate)
        high_risk_idx = int(np.argmax(y_prob))
        low_risk_idx = int(np.argmin(y_prob))
        
        # High Risk Waterfall Plot
        hr_path = os.path.join(patient_plots_dir, 'shap_patient_high_risk.png')
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_explanation[high_risk_idx], max_display=10, show=False)
        actual_label = f"True: {y_true[high_risk_idx]}" if y_true is not None else ""
        plt.title(f'Local Patient Explanation: High-Risk Profile (Pred: {y_prob[high_risk_idx]:.3f}, {actual_label})', 
                  fontsize=11, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(hr_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved high-risk patient explanation: {hr_path}")
        
        # Low Risk Waterfall Plot
        lr_path = os.path.join(patient_plots_dir, 'shap_patient_low_risk.png')
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_explanation[low_risk_idx], max_display=10, show=False)
        actual_label = f"True: {y_true[low_risk_idx]}" if y_true is not None else ""
        plt.title(f'Local Patient Explanation: Low-Risk Profile (Pred: {y_prob[low_risk_idx]:.3f}, {actual_label})', 
                  fontsize=11, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(lr_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved low-risk patient explanation: {lr_path}")
        
    return df_importance, shap_explanation
