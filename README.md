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
This work is part of a research project supervised by **Dr. Tiehang Duan** (Grand Valley State University), with detailed theoretical background documented in the accompanying [LaTeX report](docs/EEG_Analysis_Report.pdf).

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
│   ├── 05_classification_deep_learning.ipynb # CNN, EEGNet
│   ├── 06_cross_subject_validation.ipynb  # Generalization analysis
│   └── 07_results_visualization.ipynb     # Comprehensive results
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

## 🚀 Quick Start

### 1. Download Dataset
```python
# Run in notebook or terminal
from utils.data_loader import download_bci_competition_data

download_bci_competition_data(save_dir='data/raw/')
```

Or manually download from: https://www.bbci.de/competition/iv/#datasets

### 2. Run Notebooks Sequentially
```bash
jupyter notebook notebooks/01_dataset_exploration.ipynb
```

Follow notebooks in order (01 → 07) for complete pipeline.

### 3. Quick Classification Example
```python
from utils.data_loader import load_subject_data
from utils.preprocessing import preprocess_subject
from utils.features import extract_all_features
from utils.models import train_lda_classifier

# Load and preprocess
raw, labels = load_subject_data('A01', session='T')
epochs = preprocess_subject(raw, labels)

# Extract features
features = extract_all_features(epochs)

# Train classifier
model, accuracy = train_lda_classifier(features, labels)
print(f"Accuracy: {accuracy:.2f}%")
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

### 05 - Deep Learning Classification
- EEGNet architecture
- Shallow ConvNet
- Deep ConvNet
- Custom CNN implementation
- Training strategies
- Overfitting prevention
- Performance vs traditional methods
- **Output:** Deep learning results

### 06 - Cross-Subject Validation
- Subject-independent classification
- Leave-One-Subject-Out (LOSO)
- Transfer learning approaches
- Domain adaptation
- Generalization analysis
- **Output:** Generalization performance

### 07 - Results Visualization
- Comprehensive performance metrics
- Statistical significance tests
- ROC curves and AUC scores
- Cohen's Kappa
- Information Transfer Rate (ITR)
- Publication-ready figures
- **Output:** Final results and figures

---

## 📊 Results

### Subject-Dependent Classification (Mean ± Std across 9 subjects)

| Method | Accuracy | Kappa | Notes |
|--------|----------|-------|-------|
| LDA + CSP | 72.5 ± 8.3% | 0.63 | Baseline |
| SVM (RBF) + CSP | 75.1 ± 7.9% | 0.67 | Best traditional |
| Random Forest | 70.3 ± 9.1% | 0.60 | Ensemble |
| EEGNet | 78.4 ± 6.5% | 0.71 | Best overall |
| Deep ConvNet | 76.8 ± 7.2% | 0.69 | Deep learning |

### Cross-Subject Classification (LOSO)

| Method | Accuracy | Kappa |
|--------|----------|-------|
| LDA + CSP | 58.3 ± 12.1% | 0.44 |
| EEGNet | 62.7 ± 10.8% | 0.50 |

*Note: Full results with statistical tests in notebook 07*

---

## 📚 Citation

If you use this repository in your research, please cite:

```bibtex
@misc{aroua2024bci,
  author = {Aroua, Rahma},
  title = {BCI Motor Imagery Analysis: Comprehensive Pipeline for EEG Classification},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/rahmaaroua/bci-motor-imagery-analysis},
  note = {Supervised by Dr. Tiehang Duan, Grand Valley State University}
}
```

And the original dataset:

```bibtex
@article{brunner2008bci,
  title={BCI Competition 2008--Graz data set A},
  author={Brunner, Clemens and Leeb, Robert and M{\"u}ller-Putz, Gernot and Schl{\"o}gl, Alois and Pfurtscheller, Gert},
  journal={Institute for Knowledge Discovery (Laboratory of Brain-Computer Interfaces), Graz University of Technology},
  volume={16},
  pages={1--6},
  year={2008}
}
```

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
- Email: aroua.rahma@example.com

**Supervisor: Dr. Tiehang Duan**
- Email: duant@gvsu.edu
- Institution: Grand Valley State University

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- BCI Competition IV organizers for the dataset
- Dr. Tiehang Duan for supervision and guidance
- MNE-Python community for excellent EEG analysis tools
- Grand Valley State University for research support


---

**⭐ Star this repo if you find it helpful!**
