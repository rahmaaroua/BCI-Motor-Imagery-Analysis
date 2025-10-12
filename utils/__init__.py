"""
BCI Motor Imagery Analysis Utilities

This package provides utility functions for:
- Data loading and preprocessing
- Feature extraction
- Model training and evaluation
- Visualization

Author: Rahma Aroua
Supervisor: Dr. Tiehang Duan
"""

__version__ = "1.0.0"
__author__ = "Rahma Aroua"

from .data_loader import (
    load_subject_data,
    load_all_subjects,
    download_bci_competition_data
)

from .preprocessing import (
    apply_bandpass_filter,
    apply_notch_filter,
    apply_ica,
    preprocess_subject,
    create_epochs
)

from .features import (
    extract_csp_features,
    extract_band_power,
    extract_hjorth_parameters,
    extract_statistical_features,
    extract_all_features
)

from .models import (
    train_lda_classifier,
    train_svm_classifier,
    train_random_forest,
    evaluate_model
)

from .evaluation import (
    compute_metrics,
    cross_validate_subject,
    plot_confusion_matrix,
    compute_cohens_kappa
)

from .visualization import (
    plot_raw_signals,
    plot_psd,
    plot_csp_patterns,
    plot_erds_maps,
    plot_feature_importance
)

__all__ = [
    # Data loading
    'load_subject_data',
    'load_all_subjects',
    'download_bci_competition_data',

    # Preprocessing
    'apply_bandpass_filter',
    'apply_notch_filter',
    'apply_ica',
    'preprocess_subject',
    'create_epochs',

    # Features
    'extract_csp_features',
    'extract_band_power',
    'extract_hjorth_parameters',
    'extract_statistical_features',
    'extract_all_features',

    # Models
    'train_lda_classifier',
    'train_svm_classifier',
    'train_random_forest',
    'evaluate_model',

    # Evaluation
    'compute_metrics',
    'cross_validate_subject',
    'plot_confusion_matrix',
    'compute_cohens_kappa',

    # Visualization
    'plot_raw_signals',
    'plot_psd',
    'plot_csp_patterns',
    'plot_erds_maps',
    'plot_feature_importance',
]