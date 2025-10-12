"""
Model Evaluation and Validation Utilities

Functions for computing metrics, cross-validation, and statistical tests.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, cohen_kappa_score,
                             roc_curve, auc, classification_report)
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from typing import Tuple, Dict, Optional


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    class_names: Optional[list] = None) -> Dict:
    """
    Compute comprehensive classification metrics

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities (n_samples, n_classes)
    class_names : list, optional
        Names of classes
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    if class_names is None:
        class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']

    n_classes = len(class_names)

    # Binarize labels for multi-class ROC
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=np.arange(1, n_classes + 1))

    fig, ax = plt.subplots(figsize=figsize)

    colors = ['blue', 'red', 'green', 'orange']

    for i, (color, class_name) in enumerate(zip(colors, class_names)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Multi-class Classification', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def compare_models(results_dict: Dict, metric: str = 'accuracy') -> plt.Figure:
    """
    Compare multiple models

    Parameters
    ----------
    results_dict : dict
        Dictionary with model names as keys and results as values
    metric : str
        Metric to compare ('accuracy', 'kappa', 'f1')

    Returns
    -------
    fig : matplotlib.figure.Figure
        Comparison plot
    """
    model_names = list(results_dict.keys())
    means = [results_dict[name][f'{metric}_mean'] for name in model_names]
    stds = [results_dict[name][f'{metric}_std'] for name in model_names]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(model_names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                  color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f'Model Comparison - {metric.capitalize()}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{mean:.3f}±{std:.3f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def statistical_comparison(results1: np.ndarray, results2: np.ndarray,
                           test: str = 'wilcoxon') -> Dict:
    """
    Perform statistical test to compare two models

    Parameters
    ----------
    results1 : np.ndarray
        Results from model 1 (e.g., accuracies across folds/subjects)
    results2 : np.ndarray
        Results from model 2
    test : str
        Statistical test ('wilcoxon', 'ttest', 'mannwhitney')

    Returns
    -------
    test_results : dict
        Statistical test results
    """
    if test == 'wilcoxon':
        statistic, p_value = stats.wilcoxon(results1, results2)
        test_name = "Wilcoxon Signed-Rank Test"
    elif test == 'ttest':
        statistic, p_value = stats.ttest_rel(results1, results2)
        test_name = "Paired t-test"
    elif test == 'mannwhitney':
        statistic, p_value = stats.mannwhitneyu(results1, results2)
        test_name = "Mann-Whitney U Test"
    else:
        raise ValueError(f"Unknown test: {test}")

    test_results = {
        'test_name': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'mean_diff': np.mean(results1) - np.mean(results2)
    }

    print(f"\n{test_name}:")
    print(f"  Statistic: {statistic:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant (α=0.05): {'Yes' if test_results['significant'] else 'No'}")
    print(f"  Mean difference: {test_results['mean_diff']:.4f}")

    return test_results


def compute_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence interval

    Parameters
    ----------
    data : np.ndarray
        Data array
    confidence : float
        Confidence level (default: 0.95)

    Returns
    -------
    ci_lower, ci_upper : tuple
        Lower and upper bounds of confidence interval
    """
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    ci = se * stats.t.ppf((1 + confidence) / 2., n - 1)

    return mean - ci, mean + ci


def create_performance_summary(all_metrics: Dict, save_path: Optional[str] = None) -> None:
    """
    Create and display comprehensive performance summary

    Parameters
    ----------
    all_metrics : dict
        Dictionary containing all computed metrics
    save_path : str, optional
        Path to save summary table
    """
    import pandas as pd

    summary_data = []

    for model_name, metrics in all_metrics.items():
        row = {
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}",
            'Kappa': f"{metrics['kappa_mean']:.4f} ± {metrics['kappa_std']:.4f}",
            'F1-Score': f"{metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}",
        }

        if 'itr' in metrics:
            row['ITR (bits/min)'] = f"{metrics['itr']:.2f}"

        summary_data.append(row)

    df = pd.DataFrame(summary_data)

    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"✓ Summary saved to {save_path}")

    return df
    y_pred: np.ndarray
    Predicted
    labels


class_names: list, optional
Names
of
classes
for reporting

Returns
-------
metrics: dict
Dictionary
containing
all
metrics
"""
if class_names is None:
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']

metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
    'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
    'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
    'cohen_kappa': cohen_kappa_score(y_true, y_pred),
    'confusion_matrix': confusion_matrix(y_true, y_pred),
}

# Per-class metrics
report = classification_report(y_true, y_pred, target_names=class_names, 
                               output_dict=True, zero_division=0)
metrics['per_class'] = report

return metrics


def compute_cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
"""
Compute
Cohen
's Kappa coefficient

Kappa
measures
agreement
between
predictions and ground
truth,
accounting
for chance agreement.

Parameters
----------
y_true: np.ndarray
True
labels
y_pred: np.ndarray
Predicted
labels

Returns
-------
kappa: float
Cohen
's Kappa score
"""
return cohen_kappa_score(y_true, y_pred)


def compute_information_transfer_rate(accuracy: float, n_classes: int = 4, 
                                  trial_duration: float = 4.0) -> float:
"""
Compute
Information
Transfer
Rate(ITR) in bits
per
minute

ITR
quantifies
communication
speed
of
a
BCI
system

Parameters
----------
accuracy: float
Classification
accuracy(0
to
1)
n_classes: int
Number
of
classes
trial_duration: float
Duration
of
one
trial in seconds

Returns
-------
itr: float
Information
transfer
rate in bits / minute
"""
if accuracy == 1.0:
    accuracy = 0.9999  # Avoid log(0)
elif accuracy == 0.0:
    accuracy = 0.0001

# ITR formula from Wolpaw et al.
P = accuracy
N = n_classes

if P < 1.0 / N:
    return 0.0  # Below chance level

bits_per_trial = np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1))
trials_per_minute = 60.0 / trial_duration
itr = bits_per_trial * trials_per_minute

return max(0, itr)  # ITR cannot be negative


def cross_validate_subject(model, X: np.ndarray, y: np.ndarray, 
                       cv: int = 5) -> Dict:
"""
Perform
stratified
k - fold
cross - validation

Parameters
----------
model: sklearn
estimator
Model
to
validate
X: np.ndarray
Feature
matrix
y: np.ndarray
Labels
cv: int
Number
of
folds

Returns
-------
results: dict
Cross - validation
results
with metrics per fold
"""
skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

fold_accuracies = []
fold_kappas = []
fold_f1s = []
all_y_true = []
all_y_pred = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    # Compute metrics
    acc = accuracy_score(y_val, y_pred)
    kappa = cohen_kappa_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)

    fold_accuracies.append(acc)
    fold_kappas.append(kappa)
    fold_f1s.append(f1)

    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred)

results = {
    'accuracy_mean': np.mean(fold_accuracies),
    'accuracy_std': np.std(fold_accuracies),
    'kappa_mean': np.mean(fold_kappas),
    'kappa_std': np.std(fold_kappas),
    'f1_mean': np.mean(fold_f1s),
    'f1_std': np.std(fold_f1s),
    'fold_accuracies': fold_accuracies,
    'fold_kappas': fold_kappas,
    'confusion_matrix': confusion_matrix(all_y_true, all_y_pred),
}

print(f"Cross-validation ({cv}-fold):")
print(f"  Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
print(f"  Kappa:    {results['kappa_mean']:.4f} ± {results['kappa_std']:.4f}")
print(f"  F1-score: {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")

return results


def leave_one_subject_out_validation(all_subjects_data: Dict, 
                                 model_func, feature_func) -> Dict:
"""
Leave - One - Subject - Out(LOSO)
cross - validation

Tests
model
generalization
across
subjects

Parameters
----------
all_subjects_data: dict
Dictionary
with subject data
model_func: callable
Function
that
returns
a
model
feature_func: callable
Function
that
extracts
features

Returns
-------
results: dict
LOSO
validation
results
"""
subject_ids = list(all_subjects_data.keys())
subject_accuracies = {}

print("Performing Leave-One-Subject-Out validation...")

for test_subject in subject_ids:
    print(f"\nTesting on {test_subject}...")

    # Prepare training data (all subjects except test_subject)
    X_train_list, y_train_list = [], []
    for subject_id, (raw, labels) in all_subjects_data.items():
        if subject_id != test_subject:
            features = feature_func(raw, labels)
            X_train_list.append(features)
            y_train_list.append(labels)

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)

    # Prepare test data
    raw_test, y_test = all_subjects_data[test_subject]
    X_test = feature_func(raw_test, y_test)

    # Train and evaluate
    model = model_func()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    subject_accuracies[test_subject] = {'accuracy': acc, 'kappa': kappa}
    print(f"  {test_subject}: Accuracy = {acc:.4f}, Kappa = {kappa:.4f}")

# Aggregate results
accuracies = [v['accuracy'] for v in subject_accuracies.values()]
kappas = [v['kappa'] for v in subject_accuracies.values()]

results = {
    'subject_results': subject_accuracies,
    'mean_accuracy': np.mean(accuracies),
    'std_accuracy': np.std(accuracies),
    'mean_kappa': np.mean(kappas),
    'std_kappa': np.std(kappas),
}

print(f"\n{'='*50}")
print(f"LOSO Results:")
print(f"  Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
print(f"  Mean Kappa:    {results['mean_kappa']:.4f} ± {results['std_kappa']:.4f}")
print(f"{'='*50}")

return results


def plot_confusion_matrix(cm: np.ndarray, 
                     class_names: Optional[list] = None,
                     normalize: bool = True,
                     figsize: Tuple[int, int] = (8, 6),
                     cmap: str = 'Blues') -> plt.Figure:
"""
Plot
confusion
matrix

Parameters
----------
cm: np.ndarray
Confusion
matrix
class_names: list, optional
Names
of
classes
normalize: bool
Whether
to
normalize
figsize: tuple
Figure
size
cmap: str
Colormap

Returns
-------
fig: matplotlib.figure.Figure
Figure
object
"""
if class_names is None:
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']

fig, ax = plt.subplots(figsize=figsize)

if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fmt = '.2f'
    title = 'Normalized Confusion Matrix'
else:
    fmt = 'd'
    title = 'Confusion Matrix'

sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'},
            ax=ax)

ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title(title, fontsize=14, fontweight='bold')

plt.tight_layout()
return fig


def plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray, 
               class_names: Optional[list] = None,
               figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
"""
Plot
ROC
curves
for multi -class classification

Parameters
----------
y_true: np.ndarray
True
labels