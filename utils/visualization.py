"""
Visualization Utilities for EEG Analysis

Functions for plotting EEG signals, topographies, features, and results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from typing import Optional, List, Tuple
from mne.decoding import CSP

# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# ==================== Raw Signal Visualization ====================

def plot_raw_signals(raw: mne.io.Raw,
                     channels: Optional[List[str]] = None,
                     start: float = 50.0,
                     duration: float = 10.0,
                     figsize: Tuple[int, int] = (12, 8),
                     save_path: Optional[str] = None):
    """
    Plot raw EEG signals from specified channels

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    channels : list, optional
        Channels to plot (default: C3, Cz, C4, FC3, FC4)
    start : float
        Start time in seconds (default: 50s to avoid initial artifacts)
    duration : float
        Duration to plot in seconds (default: 10s)
    figsize : tuple
        Figure size (default: (12, 8))
    save_path : str, optional
        Path to save figure

    Examples
    --------
    >>> plot_raw_signals(raw, channels=['C3', 'Cz', 'C4'], duration=10)
    """
    if channels is None:
        channels = [ch for ch in ['C3', 'Cz', 'C4', 'FC3', 'FC4'] if ch in raw.ch_names]

    # Get data
    sfreq = raw.info['sfreq']
    start_sample = int(start * sfreq)
    stop_sample = int((start + duration) * sfreq)

    data, times = raw.get_data(
        picks=channels,
        start=start_sample,
        stop=stop_sample,
        return_times=True
    )

    # Convert to microvolts
    data = data * 1e6

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot with offset
    offset = 80  # microvolts
    colors = plt.cm.Set1(np.linspace(0, 1, len(channels)))

    for i, ch_name in enumerate(channels):
        ax.plot(times, data[i] + i * offset, label=ch_name,
                linewidth=0.8, color=colors[i])

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Channel', fontsize=12)
    ax.set_title('Raw EEG Signals - Motor Cortex Channels',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Set y-tick labels
    yticks = [i * offset for i in range(len(channels))]
    ax.set_yticks(yticks)
    ax.set_yticklabels(channels)

    # Add scale reference
    ax.text(0.02, 0.98, f'Scale: {offset}μV between channels',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to {save_path}")

    plt.show()


# ==================== PSD Visualization ====================

def plot_psd(raw: mne.io.Raw,
             channels: Optional[List[str]] = None,
             fmin: float = 1.0,
             fmax: float = 40.0,
             figsize: Tuple[int, int] = (12, 6),
             save_path: Optional[str] = None):
    """
    Plot Power Spectral Density for specified channels

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    channels : list, optional
        Channels to plot (default: C3, Cz, C4, Fz, Pz)
    fmin : float
        Minimum frequency (default: 1 Hz)
    fmax : float
        Maximum frequency (default: 40 Hz)
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Examples
    --------
    >>> plot_psd(raw, channels=['C3', 'Cz', 'C4'], fmin=1, fmax=40)
    """
    if channels is None:
        # Try to find motor channels with 'EEG-' prefix or without
        channels = []
        for ch in ['C3', 'Cz', 'C4', 'Fz', 'Pz']:
            if ch in raw.ch_names:
                channels.append(ch)
            elif f'EEG-{ch}' in raw.ch_names:
                channels.append(f'EEG-{ch}')

    # Pick channels
    raw_subset = raw.copy().pick_channels(channels)

    # Compute PSD
    psd = raw_subset.compute_psd(fmin=fmin, fmax=fmax, n_fft=1024, verbose=False)
    psd_data = psd.get_data()
    freqs = psd.freqs

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set1(np.linspace(0, 1, len(channels)))

    for i, ch in enumerate(channels):
        ax.semilogy(freqs, psd_data[i], label=ch, linewidth=2, color=colors[i])

    # Highlight frequency bands
    ax.axvspan(8, 12, alpha=0.2, color='blue', label='Mu (8-12 Hz)')
    ax.axvspan(13, 30, alpha=0.2, color='red', label='Beta (13-30 Hz)')

    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Power Spectral Density (μV²/Hz)', fontsize=12)
    ax.set_title('Power Spectral Density - Motor Cortex Channels',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to {save_path}")

    plt.show()


# ==================== Electrode Layout ====================

def plot_electrode_positions(raw: mne.io.Raw,
                             figsize: Tuple[int, int] = (8, 8),
                             save_path: Optional[str] = None):
    """
    Plot electrode positions on scalp

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data with montage set
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Get EEG channels
    picks_eeg = mne.pick_types(raw.info, eeg=True, eog=False)
    eeg_ch_names = [raw.ch_names[p] for p in picks_eeg]

    # Dummy data for visualization
    dummy_data = np.ones(len(picks_eeg))

    try:
        # Plot topography
        im, _ = mne.viz.plot_topomap(
            dummy_data, raw.info, picks=picks_eeg,
            axes=ax, show=False, contours=0,
            names=eeg_ch_names, show_names=True,
            sphere='auto'
        )

        ax.set_title('EEG Electrode Positions (10-20 System)',
                     fontsize=14, fontweight='bold', pad=20)

        # Highlight motor cortex
        motor_chs = [ch for ch in ['C3', 'Cz', 'C4'] if ch in raw.ch_names]
        if motor_chs:
            ax.text(0.02, 0.02, f'Motor channels: {", ".join(motor_chs)}',
                    transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    except Exception as e:
        print(f"Topographic plot failed: {e}")
        raw.plot_sensors(show_names=True, show=False, axes=ax)
        ax.set_title('EEG Electrode Positions', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to {save_path}")

    plt.show()


# ==================== Class Distribution ====================

def plot_class_distribution(labels: np.ndarray,
                            class_names: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (8, 6),
                            save_path: Optional[str] = None):
    """
    Plot class distribution bar chart

    Parameters
    ----------
    labels : np.ndarray
        Class labels
    class_names : list, optional
        Names of classes (default: Left Hand, Right Hand, Feet, Tongue)
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    if class_names is None:
        class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']

    # Count occurrences
    unique_labels = np.unique(labels)
    counts = [np.sum(labels == label) for label in unique_labels]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(class_names, counts, color=sns.color_palette("Set2", len(class_names)))

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Number of Trials', fontsize=12)
    ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to {save_path}")

    plt.show()


# ==================== CSP Patterns ====================

def plot_csp_patterns(csp: CSP,
                      info: mne.Info,
                      n_patterns: int = 6,
                      figsize: Tuple[int, int] = (15, 5),
                      save_path: Optional[str] = None):
    """
    Plot CSP spatial patterns as topographic maps

    Parameters
    ----------
    csp : CSP
        Fitted CSP object
    info : mne.Info
        MNE info object with channel information
    n_patterns : int
        Number of patterns to plot (default: 6)
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    # Get CSP patterns
    patterns = csp.patterns_

    # Plot first 3 and last 3 patterns
    indices = list(range(3)) + list(range(-3, 0))

    for idx, ax in enumerate(axes):
        pattern_idx = indices[idx]

        mne.viz.plot_topomap(
            patterns[pattern_idx],
            info,
            axes=ax,
            show=False,
            contours=6,
            cmap='RdBu_r'
        )

        if pattern_idx >= 0:
            ax.set_title(f'CSP Component {pattern_idx + 1}', fontweight='bold')
        else:
            ax.set_title(f'CSP Component {patterns.shape[0] + pattern_idx + 1}',
                         fontweight='bold')

    plt.suptitle('Common Spatial Patterns', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to {save_path}")

    plt.show()


# ==================== Feature Importance ====================

def plot_feature_importance(feature_names: List[str],
                            importances: np.ndarray,
                            top_k: int = 20,
                            figsize: Tuple[int, int] = (10, 8),
                            save_path: Optional[str] = None):
    """
    Plot feature importance bar chart

    Parameters
    ----------
    feature_names : list
        Names of features
    importances : np.ndarray
        Importance scores
    top_k : int
        Number of top features to plot
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    # Get top k features
    top_indices = np.argsort(importances)[-top_k:][::-1]
    top_names = [feature_names[i] for i in top_indices]
    top_scores = importances[top_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_scores, color=sns.color_palette("viridis", len(top_names)))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top {top_k} Most Discriminative Features',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to {save_path}")

    plt.show()


# ==================== ERD/ERS Maps ====================

def plot_erds_maps(erds_dict: dict,
                   figsize: Tuple[int, int] = (12, 10),
                   save_path: Optional[str] = None):
    """
    Plot ERD/ERS time-frequency maps for all classes

    Parameters
    ----------
    erds_dict : dict
        Dictionary with ERD/ERS data for each class
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    n_classes = len(erds_dict)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']

    for idx, (key, power) in enumerate(erds_dict.items()):
        ax = axes[idx]

        # Average over channels (or specific channels)
        power_avg = power.average(picks=['C3', 'Cz', 'C4'])

        # Plot
        im = power_avg.plot(
            axes=ax,
            show=False,
            colorbar=False,
            cmap='RdBu_r',
            vmin=-50,
            vmax=50
        )

        ax.set_title(class_names[idx], fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Frequency (Hz)', fontsize=10)

    # Add colorbar
    cbar = fig.colorbar(im[0], ax=axes, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('ERD/ERS (%)', fontsize=12)

    plt.suptitle('Event-Related Desynchronization/Synchronization',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to {save_path}")

    plt.show()


# ==================== Channel Amplitude ====================

def plot_channel_amplitude(raw: mne.io.Raw,
                           figsize: Tuple[int, int] = (10, 5),
                           save_path: Optional[str] = None):
    """
    Plot mean amplitude per EEG channel

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    # Get EEG data in microvolts
    picks_eeg = mne.pick_types(raw.info, eeg=True, eog=False)
    data_uV = raw.get_data(picks=picks_eeg) * 1e6
    eeg_ch_names = [raw.ch_names[p] for p in picks_eeg]

    # Compute mean absolute amplitude
    mean_amp = np.mean(np.abs(data_uV), axis=1)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.viridis(np.linspace(0, 1, len(mean_amp)))
    bars = ax.bar(range(len(mean_amp)), mean_amp, color=colors)

    ax.set_xticks(range(len(mean_amp)))
    ax.set_xticklabels(eeg_ch_names, rotation=90)
    ax.set_xlabel('Channel', fontsize=12)
    ax.set_ylabel('Mean |Amplitude| (μV)', fontsize=12)
    ax.set_title('Mean Amplitude per EEG Channel', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to {save_path}")

    plt.show()

# ==================== Plot csp robust ====================
import matplotlib.pyplot as plt
import numpy as np
import mne

def plot_csp_patterns_robust(csp, epochs, save_path, n_components=4):
    """
    Plot first m and last m CSP spatial patterns (robust version).
    Saves figure to save_path.
    """
    patterns = csp.patterns_
    m = n_components // 2

    first_components = patterns[:, :m]
    last_components = patterns[:, -m:]

    fig, axes = plt.subplots(2, m, figsize=(4 * m, 7))

    for i in range(m):
        ax = axes[0, i]
        try:
            mne.viz.plot_topomap(first_components[:, i], epochs.info, axes=ax,
                                 show=False, cmap="RdBu_r", contours=6)
            ax.set_title(f"Component {i+1}")
        except Exception:
            ax.bar(np.arange(patterns.shape[0]), first_components[:, i])

    for i in range(m):
        ax = axes[1, i]
        try:
            mne.viz.plot_topomap(last_components[:, i], epochs.info, axes=ax,
                                 show=False, cmap="RdBu_r", contours=6)
            ax.set_title(f"Component {n_components - m + i + 1}")
        except Exception:
            ax.bar(np.arange(patterns.shape[0]), last_components[:, i])

    fig.suptitle("CSP Spatial Patterns", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0.03, 0, 1, 0.96])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ==================== Summary Statistics ====================

def print_data_summary(raw: mne.io.Raw, labels: np.ndarray):
    """
    Print summary statistics for LaTeX table

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    labels : np.ndarray
        Class labels
    """
    # Get EEG data in microvolts
    picks_eeg = mne.pick_types(raw.info, eeg=True, eog=False)
    data_uV = raw.get_data(picks=picks_eeg) * 1e6

    # Calculate statistics
    mean_amp = np.mean(np.abs(data_uV))
    std_amp = np.std(np.abs(data_uV))
    min_amp = np.min(np.abs(data_uV))
    max_amp = np.max(np.abs(data_uV))

    # Get metadata
    n_trials = len(labels)
    n_eeg_channels = len(picks_eeg)
    n_eog_channels = len(mne.pick_types(raw.info, eog=True))
    sampling_rate = raw.info['sfreq']
    duration_sec = raw.times[-1]
    unique_classes = len(np.unique(labels))

    print("\n" + "=" * 60)
    print("DATASET SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Sampling Rate:      {sampling_rate:.0f} Hz")
    print(f"Number of Trials:   {n_trials}")
    print(f"EEG Channels:       {n_eeg_channels}")
    print(f"EOG Channels:       {n_eog_channels}")
    print(f"Mean Amplitude:     {mean_amp:.1f} ± {std_amp:.1f} μV")
    print(f"Amplitude Range:    {min_amp:.1f} - {max_amp:.1f} μV")
    print(f"Recording Duration: {duration_sec:.1f} seconds ({duration_sec / 60:.1f} minutes)")
    print(f"Classes:            {unique_classes}")
    print("=" * 60 + "\n")