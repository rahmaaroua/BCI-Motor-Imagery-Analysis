"""
Feature Extraction Utilities

Functions for extracting spatial, spectral, temporal, and statistical features from EEG data.
"""

import numpy as np
import mne
from mne.decoding import CSP
from scipy import signal, stats
from typing import Tuple, Optional, Dict, List
from sklearn.preprocessing import StandardScaler


# ==================== Common Spatial Patterns (CSP) ====================

def extract_csp_features(epochs: mne.Epochs,
                         labels: np.ndarray,
                         n_components: int = 6,
                         reg: Optional[str] = None,
                         log: bool = True) -> Tuple[np.ndarray, CSP]:
    """
    Extract Common Spatial Pattern features

    CSP finds spatial filters that maximize variance for one class
    while minimizing it for another class.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    labels : np.ndarray
        Class labels
    n_components : int
        Number of CSP components to use (default: 6, means first 3 and last 3)
    reg : str, optional
        Regularization type ('ledoit_wolf', 'oas', or None)
    log : bool
        Apply log transform to variance (default: True)

    Returns
    -------
    csp_features : np.ndarray
        CSP feature matrix (n_trials, n_components)
    csp : CSP
        Fitted CSP object

    Examples
    --------
    >>> csp_features, csp = extract_csp_features(epochs, labels, n_components=6)
    >>> print(f"CSP features shape: {csp_features.shape}")
    """
    print(f"Extracting CSP features (n_components={n_components})...")

    # Initialize CSP
    csp = CSP(
        n_components=n_components,
        reg=reg,
        log=log,
        norm_trace=False
    )

    # Get data
    X = epochs.get_data()

    # Fit and transform
    csp_features = csp.fit_transform(X, labels)

    print(f"✓ CSP features extracted: {csp_features.shape}")
    return csp_features, csp


# ==================== Band Power Features ====================

def extract_band_power(epochs: mne.Epochs,
                       bands: Dict[str, Tuple[float, float]] = None,
                       method: str = 'welch') -> np.ndarray:
    """
    Extract band power features from EEG epochs

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    bands : dict
        Frequency bands as {'band_name': (fmin, fmax)}
        Default: {'mu': (8, 12), 'beta': (13, 30)}
    method : str
        PSD computation method ('welch' or 'multitaper')

    Returns
    -------
    band_powers : np.ndarray
        Band power features (n_trials, n_channels * n_bands)

    Examples
    --------
    >>> bands = {'mu': (8, 12), 'beta': (13, 30)}
    >>> band_powers = extract_band_power(epochs, bands=bands)
    """
    if bands is None:
        bands = {
            'mu': (8, 12),
            'beta': (13, 30),
            'low_beta': (13, 20),
            'high_beta': (20, 30)
        }

    print(f"Extracting band power features for {len(bands)} bands...")

    # Get data
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = data.shape
    sfreq = epochs.info['sfreq']

    # Compute PSD for all epochs
    psd, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=1,
        fmax=40,
        n_fft=256,
        verbose=False
    )

    # Extract band powers
    band_powers_list = []

    for band_name, (fmin, fmax) in bands.items():
        # Find frequency indices
        freq_mask = (freqs >= fmin) & (freqs <= fmax)

        # Compute mean power in band for each epoch and channel
        band_power = np.mean(psd[:, :, freq_mask], axis=2)

        # Flatten to (n_epochs, n_channels)
        band_powers_list.append(band_power)

    # Stack all bands: (n_epochs, n_channels * n_bands)
    band_powers = np.hstack(band_powers_list)

    print(f"✓ Band power features extracted: {band_powers.shape}")
    return band_powers


# ==================== Hjorth Parameters ====================

def compute_hjorth_parameters(data: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute Hjorth parameters for a single channel time series

    Parameters
    ----------
    data : np.ndarray
        Time series data (1D array)

    Returns
    -------
    activity : float
        Variance of the signal
    mobility : float
        Square root of variance of the first derivative / variance
    complexity : float
        Mobility of the first derivative / mobility of the signal
    """
    # Activity (variance)
    activity = np.var(data)

    # First derivative
    diff1 = np.diff(data)

    # Mobility
    mobility = np.sqrt(np.var(diff1) / activity)

    # Second derivative
    diff2 = np.diff(diff1)

    # Complexity
    mobility_diff = np.sqrt(np.var(diff2) / np.var(diff1))
    complexity = mobility_diff / mobility

    return activity, mobility, complexity


def extract_hjorth_parameters(epochs: mne.Epochs) -> np.ndarray:
    """
    Extract Hjorth parameters for all epochs and channels

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data

    Returns
    -------
    hjorth_features : np.ndarray
        Hjorth features (n_trials, n_channels * 3)
        3 parameters: activity, mobility, complexity

    Examples
    --------
    >>> hjorth = extract_hjorth_parameters(epochs)
    >>> print(f"Hjorth features shape: {hjorth.shape}")
    """
    print("Extracting Hjorth parameters...")

    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = data.shape

    hjorth_features = np.zeros((n_epochs, n_channels * 3))

    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            activity, mobility, complexity = compute_hjorth_parameters(
                data[epoch_idx, ch_idx, :]
            )

            hjorth_features[epoch_idx, ch_idx * 3] = activity
            hjorth_features[epoch_idx, ch_idx * 3 + 1] = mobility
            hjorth_features[epoch_idx, ch_idx * 3 + 2] = complexity

    print(f"✓ Hjorth parameters extracted: {hjorth_features.shape}")
    return hjorth_features


# ==================== Statistical Features ====================

def extract_statistical_features(epochs: mne.Epochs) -> np.ndarray:
    """
    Extract statistical features from epochs

    Features: mean, std, skewness, kurtosis for each channel

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data

    Returns
    -------
    stat_features : np.ndarray
        Statistical features (n_trials, n_channels * 4)

    Examples
    --------
    >>> stat_features = extract_statistical_features(epochs)
    """
    print("Extracting statistical features...")

    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = data.shape

    stat_features = np.zeros((n_epochs, n_channels * 4))

    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            signal_data = data[epoch_idx, ch_idx, :]

            # Mean
            stat_features[epoch_idx, ch_idx * 4] = np.mean(signal_data)

            # Standard deviation
            stat_features[epoch_idx, ch_idx * 4 + 1] = np.std(signal_data)

            # Skewness
            stat_features[epoch_idx, ch_idx * 4 + 2] = stats.skew(signal_data)

            # Kurtosis
            stat_features[epoch_idx, ch_idx * 4 + 3] = stats.kurtosis(signal_data)

    print(f"✓ Statistical features extracted: {stat_features.shape}")
    return stat_features


# ==================== Combined Feature Extraction ====================

def extract_all_features(epochs: mne.Epochs,
                         labels: np.ndarray,
                         include_csp: bool = True,
                         include_band_power: bool = True,
                         include_hjorth: bool = True,
                         include_stats: bool = True,
                         normalize: bool = True) -> Dict:
    """
    Extract all features from epochs

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    labels : np.ndarray
        Class labels
    include_csp : bool
        Extract CSP features (default: True)
    include_band_power : bool
        Extract band power features (default: True)
    include_hjorth : bool
        Extract Hjorth parameters (default: True)
    include_stats : bool
        Extract statistical features (default: True)
    normalize : bool
        Apply z-score normalization (default: True)

    Returns
    -------
    features_dict : dict
        Dictionary containing:
        - 'features': Combined feature matrix
        - 'labels': Class labels
        - 'feature_names': List of feature names
        - 'csp': CSP object (if included)
        - 'scaler': StandardScaler object (if normalized)

    Examples
    --------
    >>> features_dict = extract_all_features(epochs, labels)
    >>> X = features_dict['features']
    >>> y = features_dict['labels']
    """
    print("=" * 60)
    print("Extracting all features")
    print("=" * 60)

    feature_list = []
    feature_names = []
    csp_object = None

    # CSP features
    if include_csp:
        csp_features, csp_object = extract_csp_features(epochs, labels, n_components=6)
        feature_list.append(csp_features)
        feature_names.extend([f'CSP_{i + 1}' for i in range(csp_features.shape[1])])

    # Band power features
    if include_band_power:
        band_powers = extract_band_power(epochs)
        feature_list.append(band_powers)

        bands = ['mu', 'beta', 'low_beta', 'high_beta']
        ch_names = epochs.ch_names
        for band in bands:
            feature_names.extend([f'{ch}_{band}' for ch in ch_names])

    # Hjorth parameters
    if include_hjorth:
        hjorth = extract_hjorth_parameters(epochs)
        feature_list.append(hjorth)

        params = ['activity', 'mobility', 'complexity']
        ch_names = epochs.ch_names
        for param in params:
            feature_names.extend([f'{ch}_{param}' for ch in ch_names])

    # Statistical features
    if include_stats:
        stats_features = extract_statistical_features(epochs)
        feature_list.append(stats_features)

        stats_names = ['mean', 'std', 'skew', 'kurt']
        ch_names = epochs.ch_names
        for stat in stats_names:
            feature_names.extend([f'{ch}_{stat}' for ch in ch_names])

    # Combine all features
    features = np.hstack(feature_list)

    print(f"\n{'=' * 60}")
    print(f"Feature extraction complete:")
    print(f"  Total features: {features.shape[1]}")
    print(f"  Number of trials: {features.shape[0]}")

    if include_csp:
        print(f"  - CSP: {csp_features.shape[1]} features")
    if include_band_power:
        print(f"  - Band power: {band_powers.shape[1]} features")
    if include_hjorth:
        print(f"  - Hjorth: {hjorth.shape[1]} features")
    if include_stats:
        print(f"  - Statistical: {stats_features.shape[1]} features")
    print(f"{'=' * 60}")

    # Normalize features
    scaler = None
    if normalize:
        print("\nApplying z-score normalization...")
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        print("✓ Features normalized")

    # Create output dictionary
    features_dict = {
        'features': features,
        'labels': labels,
        'feature_names': feature_names,
        'csp': csp_object,
        'scaler': scaler,
        'n_features': features.shape[1]
    }

    return features_dict


# ==================== Feature Importance ====================

def compute_mutual_information(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute mutual information between features and labels

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Labels

    Returns
    -------
    mi_scores : np.ndarray
        Mutual information scores for each feature
    """
    from sklearn.feature_selection import mutual_info_classif

    print("Computing mutual information scores...")
    mi_scores = mutual_info_classif(X, y, random_state=42)

    print(f"✓ Mutual information computed for {len(mi_scores)} features")
    return mi_scores


def get_top_features(X: np.ndarray,
                     y: np.ndarray,
                     feature_names: List[str],
                     top_k: int = 20) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Get top k most important features based on mutual information

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    feature_names : list
        Names of features
    top_k : int
        Number of top features to return

    Returns
    -------
    top_indices : np.ndarray
        Indices of top features
    top_names : list
        Names of top features
    top_scores : np.ndarray
        MI scores of top features
    """
    mi_scores = compute_mutual_information(X, y)

    # Get top k indices
    top_indices = np.argsort(mi_scores)[-top_k:][::-1]
    top_names = [feature_names[i] for i in top_indices]
    top_scores = mi_scores[top_indices]

    print(f"\nTop {top_k} most discriminative features:")
    for i, (name, score) in enumerate(zip(top_names[:10], top_scores[:10]), 1):
        print(f"  {i}. {name}: {score:.4f}")

    return top_indices, top_names, top_scores


# ==================== Time-Frequency Features ====================

def compute_erds(epochs: mne.Epochs,
                 labels: np.ndarray,
                 freqs: np.ndarray = None,
                 baseline: Tuple[float, float] = (-0.5, 0)) -> Dict:
    """
    Compute Event-Related Desynchronization/Synchronization (ERD/ERS)

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data
    labels : np.ndarray
        Class labels
    freqs : np.ndarray, optional
        Frequencies to analyze (default: 8-30 Hz)
    baseline : tuple
        Baseline period (default: (-0.5, 0))

    Returns
    -------
    erds_dict : dict
        Dictionary with ERD/ERS maps for each class
    """
    if freqs is None:
        freqs = np.arange(8, 31, 1)  # 8-30 Hz

    print(f"Computing ERD/ERS maps for {len(np.unique(labels))} classes...")

    # Compute time-frequency representation
    power = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=freqs / 2.,
        return_itc=False,
        average=False,
        verbose=False
    )

    # Apply baseline correction
    power.apply_baseline(baseline=baseline, mode='percent', verbose=False)

    # Separate by class
    erds_dict = {}
    for class_label in np.unique(labels):
        class_mask = labels == class_label
        class_power = power[class_mask].average()
        erds_dict[f'class_{class_label}'] = class_power

    print("✓ ERD/ERS maps computed")
    return erds_dict