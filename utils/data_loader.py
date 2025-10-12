"""
Data loading utilities for BCI Competition IV Dataset 2a

Functions for loading raw EEG data and labels from .gdf and .mat files.
FIXED: Added workaround for NumPy/MNE compatibility issue
"""

import os
import mne
import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Tuple, Optional, List
import requests
from tqdm import tqdm


def download_bci_competition_data(save_dir: str = 'data/raw/', subjects: Optional[List[str]] = None):
    """
    Download BCI Competition IV Dataset 2a

    Parameters
    ----------
    save_dir : str
        Directory to save downloaded files
    subjects : list of str, optional
        List of subject IDs to download (e.g., ['A01', 'A02'])
        If None, downloads all 9 subjects

    Note
    ----
    Dataset URL: https://www.bbci.de/competition/iv/#datasets
    Manual download may be required due to access restrictions
    """
    os.makedirs(save_dir, exist_ok=True)

    if subjects is None:
        subjects = [f'A0{i}' for i in range(1, 10)]

    print("BCI Competition IV Dataset 2a must be downloaded manually from:")
    print("https://www.bbci.de/competition/iv/#datasets")
    print(f"\nPlease download the following files to {save_dir}:")

    for subject in subjects:
        print(f"  - {subject}T.gdf and {subject}T.mat (training)")
        print(f"  - {subject}E.gdf and {subject}E.mat (evaluation)")

    print("\nAfter downloading, your directory structure should look like:")
    print(f"{save_dir}")
    for subject in subjects:
        print(f"  ├── {subject}T.gdf")
        print(f"  ├── {subject}T.mat")
        print(f"  ├── {subject}E.gdf")
        print(f"  └── {subject}E.mat")


def load_subject_data(subject_id: str,
                      session: str = 'T',
                      data_dir: str = 'data/raw/') -> Tuple[mne.io.Raw, np.ndarray]:
    """
    Load EEG data and labels for a specific subject and session

    FIXED: Added workaround for NumPy 1.26+ compatibility issue with MNE

    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., 'A01', 'A02', ..., 'A09')
    session : str
        Session type: 'T' for training or 'E' for evaluation
    data_dir : str
        Directory containing the raw data files

    Returns
    -------
    raw : mne.io.Raw
        Raw EEG data object
    labels : np.ndarray
        Class labels for each trial (1=left hand, 2=right hand, 3=feet, 4=tongue)

    Examples
    --------
    >>> raw, labels = load_subject_data('A01', session='T')
    >>> print(f"Loaded {len(labels)} trials")
    """
    # Construct file paths
    gdf_file = os.path.join(data_dir, f'{subject_id}{session}.gdf')
    mat_file = os.path.join(data_dir, f'{subject_id}{session}.mat')

    # Check if files exist
    if not os.path.exists(gdf_file):
        raise FileNotFoundError(f"GDF file not found: {gdf_file}")
    if not os.path.exists(mat_file):
        raise FileNotFoundError(f"MAT file not found: {mat_file}")

    # WORKAROUND: Temporarily patch numpy.clip to handle the dtype issue
    # This fixes the compatibility problem between NumPy 1.26+ and MNE
    original_clip = np.clip

    def patched_clip(a, a_min, a_max, out=None, **kwargs):
        if out is not None and hasattr(out, 'dtype'):
            # If output dtype is uint32 and we're clipping with float bounds
            if out.dtype == np.uint32 and (isinstance(a_max, float) or a_max == np.inf):
                # Don't use in-place operation, convert afterward
                result = original_clip(a, a_min, np.iinfo(np.uint32).max, **kwargs)
                np.copyto(out, result.astype(np.uint32))
                return out
        return original_clip(a, a_min, a_max, out=out, **kwargs)

    # Apply patch
    np.clip = patched_clip

    try:
        # Load raw EEG data
        raw = mne.io.read_raw_gdf(gdf_file, preload=True, verbose=False)
    finally:
        # Restore original numpy.clip
        np.clip = original_clip

    # Load labels
    mat_data = sio.loadmat(mat_file)
    labels = mat_data['classlabel'].flatten()

    print(f"✓ Loaded {subject_id}{session}: {len(labels)} trials, "
          f"{raw.info['sfreq']} Hz, {len(raw.ch_names)} channels")

    return raw, labels


def load_all_subjects(session: str = 'T',
                      data_dir: str = 'data/raw/',
                      subjects: Optional[List[str]] = None) -> dict:
    """
    Load data for multiple subjects

    Parameters
    ----------
    session : str
        Session type: 'T' or 'E'
    data_dir : str
        Directory containing the raw data files
    subjects : list of str, optional
        List of subject IDs. If None, loads all 9 subjects

    Returns
    -------
    data_dict : dict
        Dictionary with subject IDs as keys and (raw, labels) tuples as values

    Examples
    --------
    >>> all_data = load_all_subjects(session='T')
    >>> for subject_id, (raw, labels) in all_data.items():
    ...     print(f"{subject_id}: {len(labels)} trials")
    """
    if subjects is None:
        subjects = [f'A0{i}' for i in range(1, 10)]

    data_dict = {}

    print(f"Loading {len(subjects)} subjects...")
    for subject_id in tqdm(subjects):
        try:
            raw, labels = load_subject_data(subject_id, session, data_dir)
            data_dict[subject_id] = (raw, labels)
        except FileNotFoundError as e:
            print(f"⚠ Skipping {subject_id}: {e}")

    print(f"✓ Successfully loaded {len(data_dict)}/{len(subjects)} subjects")

    return data_dict


def get_channel_names(eeg_only: bool = True) -> List[str]:
    """
    Get standard channel names for BCI Competition IV Dataset 2a

    Parameters
    ----------
    eeg_only : bool
        If True, return only EEG channels (excludes EOG)

    Returns
    -------
    channels : list of str
        List of channel names
    """
    all_channels = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
        'P1', 'Pz', 'P2', 'POz',
        'EOG-left', 'EOG-central', 'EOG-right'
    ]

    if eeg_only:
        return [ch for ch in all_channels if not ch.startswith('EOG')]

    return all_channels


def get_event_mapping() -> dict:
    """
    Get event ID to motor imagery task mapping

    Returns
    -------
    event_dict : dict
        Mapping from class labels to task names
    """
    return {
        1: 'left_hand',
        2: 'right_hand',
        3: 'feet',
        4: 'tongue'
    }


def get_class_distribution(labels: np.ndarray) -> dict:
    """
    Get class distribution from labels

    Parameters
    ----------
    labels : np.ndarray
        Array of class labels

    Returns
    -------
    distribution : dict
        Dictionary with class counts
    """
    event_mapping = get_event_mapping()
    unique, counts = np.unique(labels, return_counts=True)

    distribution = {}
    for class_id, count in zip(unique, counts):
        task_name = event_mapping.get(class_id, f'unknown_{class_id}')
        distribution[task_name] = count

    return distribution


def split_train_test(raw: mne.io.Raw,
                     labels: np.ndarray,
                     test_size: float = 0.2,
                     random_state: int = 42) -> Tuple:
    """
    Split data into training and testing sets

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    labels : np.ndarray
        Class labels
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    train_raw, test_raw, train_labels, test_labels
    """
    from sklearn.model_selection import train_test_split

    n_trials = len(labels)
    indices = np.arange(n_trials)

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    return train_idx, test_idx, labels[train_idx], labels[test_idx]


def save_processed_data(data: dict, filepath: str):
    """
    Save preprocessed data to disk

    Parameters
    ----------
    data : dict
        Dictionary containing processed data
    filepath : str
        Path to save file (.pkl or .h5)
    """
    import pickle

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    print(f"✓ Saved processed data to {filepath}")


def load_processed_data(filepath: str) -> dict:
    """
    Load preprocessed data from disk

    Parameters
    ----------
    filepath : str
        Path to saved file

    Returns
    -------
    data : dict
        Dictionary containing processed data
    """
    import pickle

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    print(f"✓ Loaded processed data from {filepath}")
    return data