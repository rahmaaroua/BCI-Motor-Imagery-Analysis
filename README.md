# BCI Motor Imagery Analysis
## Comprehensive Pipeline for EEG-based Motor Imagery Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MNE](https://img.shields.io/badge/MNE-1.0+-orange.svg)](https://mne.tools/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Notebooks Guide](#notebooks-guide)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## 🧠 Overview

This repository provides a complete, reproducible pipeline for EEG-based motor imagery classification using the **BCI Competition IV Dataset 2a**. The project implements state-of-the-art preprocessing, feature extraction, and classification techniques for four-class motor imagery tasks (left hand, right hand, feet, tongue).

### Key Features
- ✅ **Complete preprocessing pipeline** (filtering, ICA, artifact removal)
- ✅ **Multiple feature extraction methods** (CSP, spectral, Hjorth, statistical)
- ✅ **Comprehensive classification** (traditional ML + deep learning)
- ✅ **Interactive Jupyter notebooks** with detailed explanations
- ✅ **Reproducible results** with proper validation
- ✅ **Extensive visualizations** for interpretation

### Academic Foundation
This work is part of a research project, with detailed theoretical background documented in the accompanying [LaTeX report](docs/EEG_Analysis_Report.pdf).

---

## 📁 Repository Structure

```
bci-motor-imagery-analysis/
│
├── 📓 notebooks/                          # Interactive Jupyter notebooks
│   ├── 01_dataset_exploration.ipynb       # Data loading & visualization
│   ├── 02_preprocessing_pipeline.ipynb    # Signal preprocessing
│   ├── 03_feature_extraction.ipynb        # CSP, spectral, statistical features
│   ├── 04_classification_traditional.ipynb # LDA, SVM, Random Forest
│
├── 📦 utils/                              # Utility modules (imported in notebooks)
│   ├── __init__.py
│   ├── data_loader.py                     # Dataset loading functions
│   ├── preprocessing.py                   # Preprocessing utilities
│   ├── features.py                        # Feature extraction functions
│   ├── models.py                          # Model definitions
│   ├── evaluation.py                      # Metrics and validation
│   └── visualization.py                   # Plotting utilities
│
├── 📊 data/                               # Data directory (not tracked in git)
│   ├── raw/                               # Raw .gdf and .mat files
│   │   ├── A01T.gdf
│   │   ├── A01T.mat
│   │   └── ...
│   ├── processed/                         # Preprocessed data cache
│   └── features/                          # Extracted features cache
│
├── 🎯 models/                             # Saved models
│   ├── traditional/                       # Scikit-learn models
│   └── deep_learning/                     # PyTorch/TensorFlow models
│
├── 📈 results/                            # Generated results
│   ├── figures/                           # All generated plots
│   ├── tables/                            # CSV result tables
│   └── metrics/                           # Classification metrics
│
├── 📄 docs/                               # Documentation
│   ├── EEG_Analysis_Report.pdf            # Full academic report
│   ├── references.bib                     # Bibliography
│   └── methodology.md                     # Detailed methodology
│
├── 🔧 config/                             # Configuration files
│   ├── preprocessing_config.yaml          # Preprocessing parameters
│   ├── feature_config.yaml                # Feature extraction settings
│   └── model_config.yaml                  # Model hyperparameters
│
├── 🧪 tests/                              # Unit tests (optional)
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_models.py
│
├── 📋 requirements.txt                    # Python dependencies
├── 🐍 environment.yml                     # Conda environment
├── 🚀 setup.py                            # Package installation
├── 📖 README.md                           # This file
├── 📜 LICENSE                             # MIT License
└── 🙈 .gitignore                          # Git ignore rules
```
---

## 📓 Notebooks Guide

### 01 - Dataset Exploration
- Load BCI Competition IV Dataset 2a
- Inspect channel layout and metadata
- Visualize raw EEG signals
- Analyze class distribution
- Power spectral density analysis
- **Output:** Understanding of data structure

### 02 - Preprocessing Pipeline
- Bandpass filtering (8-30 Hz)
- Notch filtering (50 Hz power line noise)
- Independent Component Analysis (ICA)
- Artifact removal (EOG, EMG)
- Common average reference (CAR)
- Epoching and baseline correction
- **Output:** Clean, preprocessed epochs

### 03 - Feature Extraction
- Common Spatial Patterns (CSP)
- Band power features (mu, beta bands)
- Hjorth parameters (activity, mobility, complexity)
- Statistical features (mean, std, skewness, kurtosis)
- Feature importance analysis
- ERD/ERS time-frequency maps
- **Output:** Feature matrix (248 features/trial)

### 04 - Traditional Classification
- Linear Discriminant Analysis (LDA)
- Support Vector Machines (SVM)
- Random Forest
- k-Nearest Neighbors (k-NN)
- Cross-validation strategies
- Confusion matrices
- Performance comparison
- **Output:** Baseline classification results

---
## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Contact

**Rahma Aroua**
- GitHub: [@rahmaaroua](https://github.com/rahmaaroua)
- Email: rahma.aroua@etudiant-fst.utm.tn

---

## 📜 License

This project is licensed under the MIT [LICENSE](LICENSE).


---

**⭐ Star this repo if you find it helpful!**
