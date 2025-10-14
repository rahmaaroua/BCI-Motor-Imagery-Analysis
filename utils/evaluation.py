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
    y_pred : np.ndarray
        Predicted labels
    class_names : list, optional
        Names of classes for reporting

    Returns
    -------
    metrics : dict
        Dictionary containing all metrics
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
    Compute Cohen's Kappa coefficient

    Kappa measures agreement between predictions and ground truth,
    accounting for chance agreement.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels

    Returns
    -------
    kappa : float
        Cohen's Kappa score
    """
    return cohen_kappa_score(y_true, y_pred)


def compute_information_transfer_rate(accuracy: float, n_classes: int = 4,
                                      trial_duration: float = 4.0) -> float:
    """
    Compute Information Transfer Rate (ITR) in bits per minute

    ENHANCED: Better edge case handling

    Parameters
    ----------
    accuracy : float
        Classification accuracy (0 to 1)
    n_classes : int
        Number of classes
    trial_duration : float
        Duration of one trial in seconds

    Returns
    -------
    itr : float
        Information transfer rate in bits/minute

    References
    ----------
    Wolpaw, J. R., et al. (2000). Brain-computer interface technology:
    a review of the first international meeting. IEEE transactions on
    rehabilitation engineering, 8(2), 164-173.

    Examples
    --------
    >>> itr = compute_information_transfer_rate(0.75, 4, 4.0)
    >>> print(f"ITR: {itr:.2f} bits/min")
    """
    # Clip accuracy to valid range
    accuracy = np.clip(accuracy, 1e-10, 1 - 1e-10)

    P = accuracy
    N = n_classes

    # Below chance level
    if P < 1.0 / N:
        return 0.0

    # Calculate bits per trial using ITR formula
    if P >= 0.99999:
        # Near perfect accuracy
        bits_per_trial = np.log2(N)
    else:
        bits_per_trial = (
                np.log2(N) +
                P * np.log2(P) +
                (1 - P) * np.log2((1 - P) / (N - 1))
        )

    # Ensure non-negative
    bits_per_trial = max(0, bits_per_trial)

    # Convert to bits per minute
    trials_per_minute = 60.0 / trial_duration
    itr = bits_per_trial * trials_per_minute

    return max(0, itr)

def cross_validate_subject(model, X: np.ndarray, y: np.ndarray,
                           cv: int = 5) -> Dict:
    """
    Perform stratified k-fold cross-validation

    FIXED: Now properly clones model for each fold

    Parameters
    ----------
    model : sklearn estimator
        Model to validate (will be cloned for each fold)
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    cv : int
        Number of folds

    Returns
    -------
    results : dict
        Cross-validation results with metrics per fold

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> model = SVC(kernel='rbf', C=1.0)
    >>> results = cross_validate_subject(model, X, y, cv=5)
    """
    from sklearn.base import clone  # IMPORTANT: Import clone

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_kappas = []
    fold_f1s = []
    all_y_true = []
    all_y_pred = []

    print(f"\nPerforming {cv}-fold cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # FIXED: Clone model for each fold to avoid reusing fitted model
        model_fold = clone(model)

        # Train and predict
        model_fold.fit(X_train, y_train)
        y_pred = model_fold.predict(X_val)

        # Compute metrics
        acc = accuracy_score(y_val, y_pred)
        kappa = cohen_kappa_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)

        fold_accuracies.append(acc)
        fold_kappas.append(kappa)
        fold_f1s.append(f1)

        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)

        print(f"  Fold {fold}/{cv}: Acc={acc:.4f}, Kappa={kappa:.4f}, F1={f1:.4f}")

    results = {
        'accuracy_mean': np.mean(fold_accuracies),
        'accuracy_std': np.std(fold_accuracies),
        'kappa_mean': np.mean(fold_kappas),
        'kappa_std': np.std(fold_kappas),
        'f1_mean': np.mean(fold_f1s),
        'f1_std': np.std(fold_f1s),
        'fold_accuracies': fold_accuracies,
        'fold_kappas': fold_kappas,
        'fold_f1s': fold_f1s,  # ADDED: was missing
        'confusion_matrix': confusion_matrix(all_y_true, all_y_pred),
    }

    print(f"\n  Cross-validation results ({cv}-fold):")
    print(f"    Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
    print(f"    Kappa:    {results['kappa_mean']:.4f} ± {results['kappa_std']:.4f}")
    print(f"    F1-score: {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")

    return results


def plot_confusion_matrix(cm: np.ndarray,
                         class_names: Optional[list] = None,
                         normalize: bool = True,
                         figsize: Tuple[int, int] = (8, 6),
                         cmap: str = 'Blues') -> plt.Figure:
    """
    Plot confusion matrix

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix
    class_names : list, optional
        Names of classes
    normalize : bool
        Whether to normalize
    figsize : tuple
        Figure size
    cmap : str
        Colormap

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
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
    Plot ROC curves for multi-class classification

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


def create_performance_summary(all_metrics: Dict,
                               save_path: Optional[str] = None):
    """
    Create and display comprehensive performance summary

    FIXED: Now returns DataFrame (needed by notebook 04)

    Parameters
    ----------
    all_metrics : dict
        Dictionary containing all computed metrics
    save_path : str, optional
        Path to save summary table

    Returns
    -------
    df : pd.DataFrame
        Summary DataFrame

    Examples
    --------
    >>> all_results = {
    ...     'LDA': lda_cv_results,
    ...     'SVM': svm_cv_results
    ... }
    >>> df = create_performance_summary(all_results, 'summary.csv')
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


def compare_models(results_dict: Dict, metric: str = 'accuracy',
                   figsize: tuple = (12, 6)) -> plt.Figure:
    """
    Compare multiple models using box plots

    Parameters
    ----------
    results_dict : dict
        Dictionary with model names as keys and CV results as values
        Each result should have 'fold_accuracies', 'fold_kappas', etc.
    metric : str
        Metric to compare ('accuracy', 'kappa', 'f1')
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.Figure
        Comparison plot

    Examples
    --------
    >>> all_results = {
    ...     'LDA': lda_cv_results,
    ...     'SVM': svm_cv_results,
    ...     'RF': rf_cv_results
    ... }
    >>> fig = compare_models(all_results, metric='accuracy')
    >>> plt.savefig('model_comparison.png')
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract data for plotting
    model_names = list(results_dict.keys())
    data_to_plot = []

    # Map metric to the correct key in results
    if metric == 'accuracy':
        metric_key = 'fold_accuracies'
    elif metric == 'kappa':
        metric_key = 'fold_kappas'
    elif metric == 'f1':
        metric_key = 'fold_f1s'
    else:
        raise ValueError(f"Unknown metric: {metric}")

    for model_name in model_names:
        if metric_key in results_dict[model_name]:
            data_to_plot.append(results_dict[model_name][metric_key])
        else:
            print(f"Warning: {metric_key} not found for {model_name}")
            data_to_plot.append([0])  # Placeholder

    # Create box plot
    bp = ax.boxplot(data_to_plot, labels=model_names, patch_artist=True,
                    notch=True, widths=0.6)

    # Customize colors
    colors = sns.color_palette("Set2", len(model_names))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Customize whiskers and caps
    for whisker in bp['whiskers']:
        whisker.set(linewidth=1.5, linestyle='-')
    for cap in bp['caps']:
        cap.set(linewidth=1.5)
    for median in bp['medians']:
        median.set(color='red', linewidth=2)

    # Add mean markers
    means = [np.mean(data) for data in data_to_plot]
    ax.plot(range(1, len(means) + 1), means, 'D', color='darkred',
            markersize=8, label='Mean', zorder=3,
            markeredgecolor='black', markeredgewidth=1)

    # Formatting
    ax.set_ylabel(metric.capitalize(), fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Comparison: {metric.capitalize()}',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)

    # Add value annotations
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    for i, (model_name, mean_val) in enumerate(zip(model_names, means)):
        ax.text(i + 1, ax.get_ylim()[0] + y_range * 0.02,
                f'{mean_val:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    return fig