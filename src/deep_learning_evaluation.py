"""
Comprehensive Deep Learning Evaluation Suite for ICU Mortality Prediction
Computes AUROC, AUPRC, Accuracy, Precision, Recall/Sensitivity, Specificity, F1, NPV, Brier score,
optimizes classification thresholds on validation data, and generates publication-quality plots.
"""

import os
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    brier_score_loss,
)

# Apply clean publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 1.0


def calculate_dl_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Computes complete classification performance metrics for a given prediction probability array
    and decision threshold using scikit-learn.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    # Threshold-independent discrimination
    auroc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Threshold-dependent metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    sens = rec
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    return {
        "threshold": round(float(threshold), 4),
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "brier_score": round(brier, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "npv": round(npv, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "total_positives": int(tp + fn),
        "total_negatives": int(tn + fp),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    search_range: Tuple[float, float] = (0.10, 0.90),
    steps: int = 81,
) -> Tuple[float, float]:
    """
    Finds the probability threshold that maximizes F1-score on validation data.
    Strictly isolated: must NEVER be called on test sets.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).flatten()

    best_thresh = 0.50
    best_f1 = -1.0

    for thresh in np.linspace(search_range[0], search_range[1], steps):
        y_pred = (y_prob >= thresh).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh

    return round(float(best_thresh), 4), round(float(best_f1), 4)


def plot_training_histories(histories: Dict[str, Any], save_path: str):
    """
    Plots training and validation loss, AUROC, and AUPRC curves across epochs for all deep learning models.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {
        "1D CNN (48h FF)": "#1f77b4",
        "Simple RNN (48h FF)": "#ff7f0e",
        "LSTM (48h FF)": "#2ca02c",
        "LSTM (48h Simple)": "#9467bd",
        "LSTM (24h FF)": "#8c564b",
    }

    # 1. Loss
    for name, hist in histories.items():
        c = colors.get(name, "#333333")
        epochs = range(1, len(hist["loss"]) + 1)
        axes[0].plot(epochs, hist["loss"], label=f"{name} (Train)", linestyle="--", color=c, alpha=0.6)
        axes[0].plot(epochs, hist["val_loss"], label=f"{name} (Val)", linestyle="-", color=c, linewidth=1.8)
    axes[0].set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Binary Cross-Entropy Loss", fontsize=11)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. AUROC
    for name, hist in histories.items():
        c = colors.get(name, "#333333")
        epochs = range(1, len(hist.get("roc_auc", [])) + 1)
        if "val_roc_auc" in hist:
            axes[1].plot(epochs, hist["val_roc_auc"], label=f"{name} (Val)", linestyle="-", color=c, linewidth=1.8)
    axes[1].set_title("Validation AUROC Across Epochs", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("AUROC", fontsize=11)
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # 3. AUPRC
    for name, hist in histories.items():
        c = colors.get(name, "#333333")
        epochs = range(1, len(hist.get("pr_auc", [])) + 1)
        if "val_pr_auc" in hist:
            axes[2].plot(epochs, hist["val_pr_auc"], label=f"{name} (Val)", linestyle="-", color=c, linewidth=1.8)
    axes[2].set_title("Validation AUPRC Across Epochs", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch", fontsize=11)
    axes[2].set_ylabel("AUPRC", fontsize=11)
    axes[2].legend(fontsize=8, loc="lower right")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_deep_learning_roc_curves(eval_dict: Dict[str, Any], save_path: str):
    """
    Plots comparative ROC curves for XGBoost, 1D CNN, Simple RNN, and LSTM across validation and test cohorts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {
        "XGBoost (48h FF)": "#d62728",
        "1D CNN (48h FF)": "#1f77b4",
        "Simple RNN (48h FF)": "#ff7f0e",
        "LSTM (48h FF)": "#2ca02c",
        "LSTM (48h Simple)": "#9467bd",
        "LSTM (24h FF)": "#8c564b",
    }

    # Left: Validation Cohort
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.50)")
    for name, data in eval_dict.items():
        if "val" in data:
            y_true = data["val"]["y_true"]
            y_prob = data["val"]["y_prob"]
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            c = colors.get(name, "#333333")
            axes[0].plot(fpr, tpr, label=f"{name} (AUROC = {auc:.4f})", color=c, linewidth=2.0)
    axes[0].set_title("Validation Cohort ROC Curves", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    axes[0].legend(fontsize=8.5, loc="lower right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Right: Test B Cohort
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.50)")
    for name, data in eval_dict.items():
        if "test_b" in data:
            y_true = data["test_b"]["y_true"]
            y_prob = data["test_b"]["y_prob"]
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            c = colors.get(name, "#333333")
            axes[1].plot(fpr, tpr, label=f"{name} (AUROC = {auc:.4f})", color=c, linewidth=2.0)
    axes[1].set_title("Primary Held-Out Test B Cohort ROC Curves", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    axes[1].set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    axes[1].legend(fontsize=8.5, loc="lower right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_deep_learning_pr_curves(eval_dict: Dict[str, Any], save_path: str):
    """
    Plots comparative Precision-Recall curves for XGBoost, 1D CNN, Simple RNN, and LSTM.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {
        "XGBoost (48h FF)": "#d62728",
        "1D CNN (48h FF)": "#1f77b4",
        "Simple RNN (48h FF)": "#ff7f0e",
        "LSTM (48h FF)": "#2ca02c",
        "LSTM (48h Simple)": "#9467bd",
        "LSTM (24h FF)": "#8c564b",
    }

    # Left: Validation Cohort
    prev_val = 0.1388
    axes[0].axhline(y=prev_val, color="k", linestyle="--", alpha=0.5, label=f"Prevalence ({prev_val:.1%})")
    for name, data in eval_dict.items():
        if "val" in data:
            y_true = data["val"]["y_true"]
            y_prob = data["val"]["y_prob"]
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            c = colors.get(name, "#333333")
            axes[0].plot(rec, prec, label=f"{name} (AUPRC = {ap:.4f})", color=c, linewidth=2.0)
    axes[0].set_title("Validation Cohort Precision-Recall Curves", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Recall (Sensitivity)", fontsize=11)
    axes[0].set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    axes[0].legend(fontsize=8.5, loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Right: Test B Cohort
    prev_test = 0.1420
    axes[1].axhline(y=prev_test, color="k", linestyle="--", alpha=0.5, label=f"Prevalence ({prev_test:.1%})")
    for name, data in eval_dict.items():
        if "test_b" in data:
            y_true = data["test_b"]["y_true"]
            y_prob = data["test_b"]["y_prob"]
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            c = colors.get(name, "#333333")
            axes[1].plot(rec, prec, label=f"{name} (AUPRC = {ap:.4f})", color=c, linewidth=2.0)
    axes[1].set_title("Primary Held-Out Test B Cohort PR Curves", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=11)
    axes[1].set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    axes[1].legend(fontsize=8.5, loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_deep_learning_confusion_matrices(cm_dict: Dict[str, Any], save_path: str):
    """
    Plots confusion matrices for the 4 key models (XGBoost, CNN, Simple RNN, LSTM) on Test B.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))

    model_keys = ["XGBoost (48h FF)", "1D CNN (48h FF)", "Simple RNN (48h FF)", "LSTM (48h FF)"]

    for idx, model_name in enumerate(model_keys):
        ax = axes[idx]
        if model_name in cm_dict:
            cm = cm_dict[model_name]
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                ax=ax,
                annot_kws={"size": 13, "weight": "bold"},
            )
            ax.set_title(f"{model_name}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Predicted Label", fontsize=10)
            ax.set_ylabel("True Label" if idx == 0 else "", fontsize=10)
            ax.set_xticklabels(["Survive (0)", "Die (1)"])
            ax.set_yticklabels(["Survive (0)", "Die (1)"] if idx == 0 else ["", ""])
        else:
            ax.axis("off")

    plt.suptitle("Confusion Matrices on Primary Held-Out Test B (Validation-Tuned Threshold)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
