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
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

def compute_classification_metrics(y_true, y_prob, threshold=0.5):
    """
    Computes a comprehensive suite of binary classification metrics.
    
    Parameters:
        y_true: array-like of shape (N,) - binary true labels (0 or 1)
        y_prob: array-like of shape (N,) - predicted probabilities for positive class
        threshold: float - decision threshold for binary classification
        
    Returns:
        metrics: dict of evaluated performance metrics
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    
    # AUROC and AUPRC (threshold-independent)
    auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    auprc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    
    # Confusion matrix elements
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Threshold-dependent metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0)) # Sensitivity
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    
    return {
        'threshold': round(float(threshold), 4),
        'auroc': round(auroc, 4),
        'auprc': round(auprc, 4),
        'f1_score': round(f1, 4),
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'sensitivity': round(rec, 4),
        'specificity': round(spec, 4),
        'npv': round(npv, 4),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'total_positives': int(tp + fn),
        'total_negatives': int(tn + fp)
    }

def find_optimal_threshold(y_val_true, y_val_prob, metric='f1'):
    """
    Finds the optimal classification threshold strictly on validation data.
    
    Parameters:
        y_val_true: binary ground truth labels of validation set
        y_val_prob: predicted mortality probabilities of validation set
        metric: 'f1' for max F1-score, 'youden' for max (sensitivity + specificity - 1)
        
    Returns:
        best_threshold: float
        best_score: float
        threshold_curve: pd.DataFrame with performance across thresholds
    """
    thresholds = np.linspace(0.02, 0.98, 97)
    records = []
    
    best_th = 0.5
    best_score = -1.0
    
    for th in thresholds:
        m = compute_classification_metrics(y_val_true, y_val_prob, threshold=th)
        if metric == 'f1':
            score = m['f1_score']
        elif metric == 'youden':
            score = m['sensitivity'] + m['specificity'] - 1.0
        else:
            score = m['f1_score']
            
        m['score'] = score
        records.append(m)
        
        if score > best_score:
            best_score = score
            best_th = th
            
    df_curve = pd.DataFrame(records)
    return round(float(best_th), 4), round(float(best_score), 4), df_curve

def plot_evaluation_curves(y_true, y_prob, model_name='XGBoost', save_path='reports/evaluation_curves.png'):
    """
    Plots ROC curve and Precision-Recall curve side-by-side.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    baseline_pr = float(np.mean(y_true))
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. ROC Curve
    axes[0].plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'{model_name} (AUROC = {auroc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Chance (AUROC = 0.500)')
    axes[0].set_title(f'Receiver Operating Characteristic (ROC) — {model_name}', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('False Positive Rate (1 - Specificity)')
    axes[0].set_ylabel('True Positive Rate (Sensitivity)')
    axes[0].legend(loc='lower right', frameon=True)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # 2. PR Curve
    axes[1].plot(rec, prec, color='#d95f02', lw=2.5, label=f'{model_name} (AUPRC = {auprc:.3f})')
    axes[1].axhline(baseline_pr, color='gray', linestyle='--', label=f'Baseline Prevalance ({baseline_pr:.3f})')
    axes[1].set_title(f'Precision-Recall (PR) Curve — {model_name}', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Recall (Sensitivity)')
    axes[1].set_ylabel('Precision (Positive Predictive Value)')
    axes[1].legend(loc='upper right', frameon=True)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved evaluation curves to {save_path}")

def plot_confusion_matrices(cms, titles, save_path='reports/confusion_matrices.png'):
    """
    Plots multiple confusion matrices side-by-side.
    cms: list of 2x2 confusion matrices
    titles: list of titles
    """
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
        
    for i, (cm, title) in enumerate(zip(cms, titles)):
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i],
            xticklabels=['Survivor (0)', 'Death (1)'],
            yticklabels=['Survivor (0)', 'Death (1)'],
            annot_kws={'fontsize': 13, 'fontweight': 'bold'}
        )
        axes[i].set_title(title, fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrices to {save_path}")
