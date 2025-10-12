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

# Import only what's needed and available
try:
    from .data_loader import (
        load_subject_data,
        load_all_subjects,
        download_bci_competition_data,
        get_channel_names,
        get_event_mapping,
        get_class_distribution
    )
except ImportError as e:
    print(f"Warning: Could not import data_loader: {e}")

try:
    from .preprocessing import (
        apply_bandpass_filter,
        apply_notch_filter,
        apply_ica,
        preprocess_subject,
        create_epochs
    )
except ImportError as e:
    print(f"Warning: Could not import preprocessing: {e}")

try:
    from .features import (
        extract_csp_features,
        extract_band_power,
        extract_hjorth_parameters,
        extract_statistical_features,
        extract_all_features
    )
except ImportError as e:
    print(f"Warning: Could not import features: {e}")

try:
    from .models import (
        train_lda_classifier,
        train_svm_classifier,
        train_random_forest,
        evaluate_model
    )
except ImportError as e:
    print(f"Warning: Could not import models: {e}")

try:
    from .evaluation import (
        compute_metrics,
        cross_validate_subject,
        plot_confusion_matrix,
        compute_cohens_kappa
    )
except ImportError as e:
    print(f"Warning: Could not import evaluation: {e}")

try:
    from .visualization import (
        plot_raw_signals,
        plot_psd,
        plot_electrode_positions,
        plot_class_distribution,
        plot_channel_amplitude,
        print_data_summary
    )
except ImportError as e:
    print(f"Warning: Could not import visualization: {e}")

__all__ = [
    'load_subject_data',
    'load_all_subjects',
    'download_bci_competition_data',
    'get_channel_names',
    'get_event_mapping',
    'get_class_distribution',
    'apply_bandpass_filter',
    'apply_notch_filter',
    'apply_ica',
    'preprocess_subject',
    'create_epochs',
    'extract_csp_features',
    'extract_band_power',
    'extract_hjorth_parameters',
    'extract_statistical_features',
    'extract_all_features',
]