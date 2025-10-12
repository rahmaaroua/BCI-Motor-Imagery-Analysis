"""
EEG Signal Preprocessing Utilities

Functions for filtering, artifact removal, epoching, and signal cleaning.
"""

import numpy as np
import mne
from mne.preprocessing import ICA
from typing import Tuple, Optional, Union
import warnings


def apply_bandpass_filter(raw: mne.io.Raw,
                          lowcut: float = 8.0,
                          highcut: float = 30.0,
                          filter_order: int = 5,
                          method: str = 'iir') -> mne.io.Raw:
    """
    Apply bandpass filter to raw EEG data

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    lowcut : float
        Low cutoff frequency in Hz (default: 8.0 Hz for mu band)
    highcut : float
        High cutoff frequency in Hz (default: 30.0 Hz for beta band)
    filter_order : int
        Filter order (default: 5)
    method : str
        Filter method ('iir' or 'fir')

    Returns
    -------
    raw_filtered : mne.io.Raw
        Filtered raw data

    Examples
    --------
    >>> raw_filtered = apply_bandpass_filter(raw, lowcut=8, highcut=30)
    """
    raw_filtered = raw.copy()

    print(f"Applying bandpass filter: {lowcut}-{highcut} Hz")
    raw_filtered.filter(
        l_freq=lowcut,
        h_freq=highcut,
        method=method,
        picks='eeg',
        verbose=False
    )

    print(f"✓ Bandpass filter applied successfully")
    return raw_filtered


def apply_notch_filter(raw: mne.io.Raw,
                       freq: float = 50.0,
                       notch_widths: Optional[float] = None) -> mne.io.Raw:
    """
    Apply notch filter to remove power line noise

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    freq : float
        Frequency to remove (default: 50 Hz for Europe, use 60 Hz for US)
    notch_widths : float, optional
        Width of the notch filter

    Returns
    -------
    raw_notched : mne.io.Raw
        Notch-filtered raw data

    Examples
    --------
    >>> raw_notched = apply_notch_filter(raw, freq=50)
    """
    raw_notched = raw.copy()

    print(f"Applying notch filter at {freq} Hz")
    raw_notched.notch_filter(
        freqs=freq,
        picks='eeg',
        notch_widths=notch_widths,
        verbose=False
    )

    print(f"✓ Notch filter applied successfully")
    return raw_notched


def apply_ica(raw: mne.io.Raw,
              n_components: int = 22,
              method: str = 'fastica',
              random_state: int = 42,
              exclude_eog: bool = True) -> Tuple[mne.io.Raw, ICA]:
    """
    Apply Independent Component Analysis for artifact removal

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    n_components : int
        Number of ICA components (default: 22)
    method : str
        ICA method ('fastica', 'infomax', or 'picard')
    random_state : int
        Random seed for reproducibility
    exclude_eog : bool
        Automatically exclude EOG components

    Returns
    -------
    raw_clean : mne.io.Raw
        Cleaned raw data with artifacts removed
    ica : mne.preprocessing.ICA
        Fitted ICA object

    Examples
    --------
    >>> raw_clean, ica = apply_ica(raw, n_components=22)
    >>> ica.plot_sources(raw)  # Interactive component inspection
    """
    raw_for_ica = raw.copy()

    # Filter for ICA (recommended: 1-40 Hz)
    print("Pre-filtering for ICA (1-40 Hz)...")
    raw_for_ica.filter(l_freq=1.0, h_freq=40.0, picks='eeg', verbose=False)

    # Initialize ICA
    print(f"Fitting ICA with {n_components} components...")
    ica = ICA(
        n_components=n_components,
        method=method,
        random_state=random_state,
        max_iter=200
    )

    # Fit ICA
    ica.fit(raw_for_ica, picks='eeg', verbose=False)

    # Find EOG components automatically
    if exclude_eog:
        # Check if EOG channels exist
        eog_channels = mne.pick_types(raw.info, eog=True)

        if len(eog_channels) > 0:
            print("Detecting EOG components...")
            eog_indices, eog_scores = ica.find_bads_eog(
                raw_for_ica,
                ch_name=raw.ch_names[eog_channels[0]],
                threshold=2.5,
                verbose=False
            )

            if eog_indices:
                ica.exclude = eog_indices
                print(f"✓ Excluding {len(eog_indices)} EOG components: {eog_indices}")
            else:
                print("⚠ No EOG components found above threshold")
        else:
            print("⚠ No EOG channels found, skipping automatic detection")

    # Apply ICA to original (properly filtered) data
    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)

    print(f"✓ ICA applied successfully")
    return raw_clean, ica


def apply_common_average_reference(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Apply common average reference (CAR)

    CAR reduces spatially correlated noise by subtracting the average
    of all electrodes from each electrode.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data

    Returns
    -------
    raw_car : mne.io.Raw
        Re-referenced raw data
    """
    raw_car = raw.copy()

    print("Applying common average reference (CAR)...")
    raw_car.set_eeg_reference('average', projection=True, verbose=False)
    raw_car.apply_proj(verbose=False)

    print("✓ CAR applied successfully")
    return raw_car


def create_epochs(raw: mne.io.Raw,
                  labels: np.ndarray,
                  tmin: float = -0.5,
                  tmax: float = 4.0,
                  baseline: Tuple[float, float] = (-0.5, 0.0),
                  reject: Optional[dict] = None) -> mne.Epochs:
    """
    Create epochs around motor imagery cues

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    labels : np.ndarray
        Class labels (1=left, 2=right, 3=feet, 4=tongue)
    tmin : float
        Start time before event (default: -0.5s)
    tmax : float
        End time after event (default: 4.0s)
    baseline : tuple
        Baseline period for correction (default: (-0.5, 0))
    reject : dict, optional
        Rejection criteria for bad epochs (e.g., {'eeg': 150e-6})

    Returns
    -------
    epochs : mne.Epochs
        Epoched data

    Examples
    --------
    >>> epochs = create_epochs(raw, labels, tmin=-0.5, tmax=4.0)
    >>> print(f"Created {len(epochs)} epochs")
    """
    # Get events from annotations
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    # Filter events to get only motor imagery cues (classes 1-4)
    # Event IDs in the file might have different values
    motor_event_ids = {k: v for k, v in event_id.items() if '769' in k or '770' in k or '771' in k or '772' in k}

    if not motor_event_ids:
        # Fallback: try to find any events with values > 0
        motor_event_ids = {k: v for k, v in event_id.items() if v > 0}

    # Create mapping for better readability
    event_mapping = {
        'left_hand': 1,
        'right_hand': 2,
        'feet': 3,
        'tongue': 4
    }

    print(f"Creating epochs from {tmin}s to {tmax}s around cue onset...")
    print(f"Found {len(events)} events")

    # Create epochs
    epochs = mne.Epochs(
        raw,
        events,
        event_id=motor_event_ids,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        picks='eeg',
        preload=True,
        reject=reject,
        verbose=False
    )

    print(f"✓ Created {len(epochs)} epochs")
    print(f"  Epoch shape: {epochs.get_data().shape}")
    print(f"  (n_epochs, n_channels, n_timepoints)")

    return epochs


def preprocess_subject(raw: mne.io.Raw,
                       labels: np.ndarray,
                       lowcut: float = 8.0,
                       highcut: float = 30.0,
                       notch_freq: float = 50.0,
                       apply_ica_flag: bool = True,
                       n_components: int = 22,
                       tmin: float = -0.5,
                       tmax: float = 4.0) -> Tuple[mne.Epochs, Optional[ICA]]:
    """
    Complete preprocessing pipeline for one subject

    This is a convenience function that applies the full preprocessing
    pipeline: filtering, ICA, CAR, and epoching.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    labels : np.ndarray
        Class labels
    lowcut : float
        Low cutoff for bandpass (default: 8 Hz)
    highcut : float
        High cutoff for bandpass (default: 30 Hz)
    notch_freq : float
        Notch filter frequency (default: 50 Hz)
    apply_ica_flag : bool
        Whether to apply ICA (default: True)
    n_components : int
        Number of ICA components (default: 22)
    tmin : float
        Epoch start time (default: -0.5s)
    tmax : float
        Epoch end time (default: 4.0s)

    Returns
    -------
    epochs : mne.Epochs
        Preprocessed epoched data
    ica : mne.preprocessing.ICA or None
        ICA object (if applied)

    Examples
    --------
    >>> epochs, ica = preprocess_subject(raw, labels)
    >>> print(f"Preprocessed {len(epochs)} trials")
    """
    print("=" * 60)
    print("Starting preprocessing pipeline")
    print("=" * 60)

    # Step 1: Bandpass filter
    raw_filtered = apply_bandpass_filter(raw, lowcut, highcut)

    # Step 2: Notch filter
    raw_notched = apply_notch_filter(raw_filtered, notch_freq)

    # Step 3: ICA (optional)
    ica = None
    if apply_ica_flag:
        raw_clean, ica = apply_ica(raw_notched, n_components=n_components)
    else:
        raw_clean = raw_notched
        print("⚠ Skipping ICA")

    # Step 4: Common Average Reference
    raw_car = apply_common_average_reference(raw_clean)

    # Step 5: Epoching
    epochs = create_epochs(raw_car, labels, tmin=tmin, tmax=tmax)

    print("=" * 60)
    print("✓ Preprocessing pipeline complete!")
    print("=" * 60)

    return epochs, ica


def compute_psd_epochs(epochs: mne.Epochs,
                       fmin: float = 1.0,
                       fmax: float = 40.0,
                       method: str = 'welch') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density for epochs

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data
    fmin : float
        Minimum frequency (default: 1 Hz)
    fmax : float
        Maximum frequency (default: 40 Hz)
    method : str
        PSD method ('welch' or 'multitaper')

    Returns
    -------
    psd_data : np.ndarray
        PSD values (n_epochs, n_channels, n_freqs)
    freqs : np.ndarray
        Frequency values
    """
    print(f"Computing PSD using {method} method...")

    psd = epochs.compute_psd(
        method=method,
        fmin=fmin,
        fmax=fmax,
        verbose=False
    )

    psd_data = psd.get_data()
    freqs = psd.freqs

    print(f"✓ PSD computed: shape {psd_data.shape}")
    return psd_data, freqs


def detect_bad_epochs(epochs: mne.Epochs,
                      threshold: float = 150e-6) -> np.ndarray:
    """
    Detect bad epochs based on amplitude threshold

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data
    threshold : float
        Amplitude threshold in Volts (default: 150 µV)

    Returns
    -------
    bad_epochs : np.ndarray
        Boolean array indicating bad epochs
    """
    data = epochs.get_data()

    # Check maximum absolute amplitude per epoch
    max_amp = np.max(np.abs(data), axis=(1, 2))
    bad_epochs = max_amp > threshold

    n_bad = np.sum(bad_epochs)
    print(f"Detected {n_bad}/{len(epochs)} bad epochs (>{threshold * 1e6:.1f} µV)")

    return bad_epochs


def resample_raw(raw: mne.io.Raw, new_sfreq: float = 250.0) -> mne.io.Raw:
    """
    Resample raw data to new sampling frequency

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    new_sfreq : float
        New sampling frequency in Hz

    Returns
    -------
    raw_resampled : mne.io.Raw
        Resampled data
    """
    raw_resampled = raw.copy()

    old_sfreq = raw.info['sfreq']
    if old_sfreq == new_sfreq:
        print(f"Data already at {new_sfreq} Hz, skipping resampling")
        return raw_resampled

    print(f"Resampling from {old_sfreq} Hz to {new_sfreq} Hz...")
    raw_resampled.resample(new_sfreq, verbose=False)

    print(f"✓ Resampled to {new_sfreq} Hz")
    return raw_resampled