"""
Machine Learning Models for Motor Imagery Classification

Traditional ML and Deep Learning implementations.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from typing import Tuple, Optional, Dict
import joblib
import os


# ==================== Traditional ML Models ====================

def train_lda_classifier(X_train: np.ndarray,
                         y_train: np.ndarray,
                         solver: str = 'svd') -> LinearDiscriminantAnalysis:
    """
    Train Linear Discriminant Analysis classifier

    Parameters
    ----------
    X_train : np.ndarray
        Training features (n_samples, n_features)
    y_train : np.ndarray
        Training labels
    solver : str
        LDA solver ('svd', 'lsqr', 'eigen')

    Returns
    -------
    model : LinearDiscriminantAnalysis
        Trained LDA model

    Examples
    --------
    >>> model = train_lda_classifier(X_train, y_train)
    >>> accuracy = model.score(X_test, y_test)
    """
    model = LinearDiscriminantAnalysis(solver=solver)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    print(f"✓ LDA trained - Training accuracy: {train_acc:.4f}")

    return model


def train_svm_classifier(X_train: np.ndarray,
                         y_train: np.ndarray,
                         kernel: str = 'rbf',
                         C: float = 1.0,
                         gamma: str = 'scale') -> SVC:
    """
    Train Support Vector Machine classifier

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    kernel : str
        Kernel type ('linear', 'rbf', 'poly')
    C : float
        Regularization parameter
    gamma : str or float
        Kernel coefficient

    Returns
    -------
    model : SVC
        Trained SVM model

    Examples
    --------
    >>> model = train_svm_classifier(X_train, y_train, kernel='rbf')
    >>> y_pred = model.predict(X_test)
    """
    model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    print(f"✓ SVM ({kernel}) trained - Training accuracy: {train_acc:.4f}")

    return model


def train_random_forest(X_train: np.ndarray,
                        y_train: np.ndarray,
                        n_estimators: int = 100,
                        max_depth: Optional[int] = None) -> RandomForestClassifier:
    """
    Train Random Forest classifier

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    n_estimators : int
        Number of trees
    max_depth : int, optional
        Maximum tree depth

    Returns
    -------
    model : RandomForestClassifier
        Trained Random Forest model

    Examples
    --------
    >>> model = train_random_forest(X_train, y_train, n_estimators=100)
    >>> feature_importance = model.feature_importances_
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    print(f"✓ Random Forest trained - Training accuracy: {train_acc:.4f}")

    return model


def train_knn_classifier(X_train: np.ndarray,
                         y_train: np.ndarray,
                         n_neighbors: int = 5) -> KNeighborsClassifier:
    """
    Train k-Nearest Neighbors classifier

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    n_neighbors : int
        Number of neighbors

    Returns
    -------
    model : KNeighborsClassifier
        Trained k-NN model

    Examples
    --------
    >>> model = train_knn_classifier(X_train, y_train, n_neighbors=5)
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    print(f"✓ k-NN (k={n_neighbors}) trained - Training accuracy: {train_acc:.4f}")

    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Evaluate model performance on test set

    Parameters
    ----------
    model : sklearn estimator
        Trained model
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels

    Returns
    -------
    metrics : dict
        Dictionary containing evaluation metrics

    Examples
    --------
    >>> metrics = evaluate_model(model, X_test, y_test)
    >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    """
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

    print(f"Test Accuracy: {metrics['accuracy']:.4f}")

    return metrics


def cross_validate_model(model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict:
    """
    Perform cross-validation

    Parameters
    ----------
    model : sklearn estimator
        Model to validate
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    cv : int
        Number of folds

    Returns
    -------
    results : dict
        Cross-validation results

    Examples
    --------
    >>> results = cross_validate_model(model, X, y, cv=5)
    >>> print(f"Mean accuracy: {results['mean_accuracy']:.4f}")
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)

    results = {
        'mean_accuracy': scores.mean(),
        'std_accuracy': scores.std(),
        'all_scores': scores
    }

    print(f"Cross-validation: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")

    return results


def save_model(model, filepath: str):
    """
    Save trained model to disk

    Parameters
    ----------
    model : sklearn estimator
        Trained model
    filepath : str
        Path to save model

    Examples
    --------
    >>> save_model(model, 'models/traditional/lda_model.pkl')
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    joblib.dump(model, filepath)
    print(f"✓ Model saved to {filepath}")


def load_model(filepath: str):
    """
    Load trained model from disk

    Parameters
    ----------
    filepath : str
        Path to saved model

    Returns
    -------
    model : sklearn estimator
        Loaded model

    Examples
    --------
    >>> model = load_model('models/traditional/lda_model.pkl')
    """
    model = joblib.load(filepath)
    print(f"✓ Model loaded from {filepath}")
    return model


# ==================== Deep Learning Models ====================

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠ PyTorch not available. Deep learning models disabled.")

if TORCH_AVAILABLE:

    class EEGDataset(Dataset):
        """PyTorch Dataset for EEG data"""

        def __init__(self, X, y):
            """
            Parameters
            ----------
            X : np.ndarray
                EEG data (n_samples, n_channels, n_timepoints)
            y : np.ndarray
                Labels (n_samples,)
            """
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y) - 1  # Convert to 0-indexed

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


    class EEGNet(nn.Module):
        """
        EEGNet: Compact Convolutional Neural Network for EEG-based BCIs

        Reference:
        Lawhern et al. (2018). EEGNet: a compact convolutional neural network
        for EEG-based brain-computer interfaces. Journal of Neural Engineering.

        Parameters
        ----------
        n_channels : int
            Number of EEG channels
        n_classes : int
            Number of output classes
        n_timepoints : int
            Number of time points per trial
        F1 : int
            Number of temporal filters
        D : int
            Depth multiplier for depthwise convolution
        F2 : int
            Number of pointwise filters
        dropout : float
            Dropout rate
        """

        def __init__(self, n_channels=22, n_classes=4, n_timepoints=1000,
                     F1=8, D=2, F2=16, dropout=0.5):
            super(EEGNet, self).__init__()

            self.n_channels = n_channels
            self.n_classes = n_classes

            # Block 1: Temporal convolution
            self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
            self.batchnorm1 = nn.BatchNorm2d(F1)

            # Block 2: Depthwise convolution
            self.conv2 = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
            self.batchnorm2 = nn.BatchNorm2d(F1 * D)
            self.activation1 = nn.ELU()
            self.pooling1 = nn.AvgPool2d((1, 4))
            self.dropout1 = nn.Dropout(dropout)

            # Block 3: Separable convolution
            self.conv3 = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
            self.batchnorm3 = nn.BatchNorm2d(F2)
            self.activation2 = nn.ELU()
            self.pooling2 = nn.AvgPool2d((1, 8))
            self.dropout2 = nn.Dropout(dropout)

            # Calculate output size
            self.flatten_size = self._get_flatten_size(n_timepoints)

            # Classifier
            self.fc = nn.Linear(self.flatten_size, n_classes)

        def _get_flatten_size(self, n_timepoints):
            """Calculate flattened size after convolutions"""
            x = torch.zeros(1, 1, self.n_channels, n_timepoints)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.pooling1(x)
            x = self.conv3(x)
            x = self.pooling2(x)
            return x.view(1, -1).size(1)

        def forward(self, x):
            # Input shape: (batch, channels, timepoints)
            x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, channels, timepoints)

            # Block 1
            x = self.conv1(x)
            x = self.batchnorm1(x)

            # Block 2
            x = self.conv2(x)
            x = self.batchnorm2(x)
            x = self.activation1(x)
            x = self.pooling1(x)
            x = self.dropout1(x)

            # Block 3
            x = self.conv3(x)
            x = self.batchnorm3(x)
            x = self.activation2(x)
            x = self.pooling2(x)
            x = self.dropout2(x)

            # Classifier
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x


    def train_eegnet(X_train, y_train, X_val, y_val,
                     n_channels=22, n_classes=4,
                     epochs=100, batch_size=32, lr=0.001,
                     device='cpu', verbose=True):
        """
        Train EEGNet model

        Parameters
        ----------
        X_train : np.ndarray
            Training data (n_samples, n_channels, n_timepoints)
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray
            Validation data
        y_val : np.ndarray
            Validation labels
        n_channels : int
            Number of EEG channels
        n_classes : int
            Number of classes
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        lr : float
            Learning rate
        device : str
            Device ('cpu' or 'cuda')
        verbose : bool
            Print training progress

        Returns
        -------
        model : EEGNet
            Trained model
        history : dict
            Training history

        Examples
        --------
        >>> model, history = train_eegnet(X_train, y_train, X_val, y_val)
        >>> print(f"Best val accuracy: {max(history['val_acc']):.4f}")
        """
        # Create datasets
        train_dataset = EEGDataset(X_train, y_train)
        val_dataset = EEGDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Initialize model
        n_timepoints = X_train.shape[2]
        model = EEGNet(n_channels=n_channels, n_classes=n_classes,
                       n_timepoints=n_timepoints).to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training history
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        print(f"Training EEGNet for {epochs} epochs...")

        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss, train_correct = 0, 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_correct += (outputs.argmax(1) == y_batch).sum().item()

            # Validation
            model.eval()
            val_loss, val_correct = 0, 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)

                    val_loss += loss.item()
                    val_correct += (outputs.argmax(1) == y_batch).sum().item()

            # Record history
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_correct / len(train_dataset))
            history['val_loss'].append(val_loss / len(val_loader))
            history['val_acc'].append(val_correct / len(val_dataset))

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Train Loss: {history['train_loss'][-1]:.4f}, "
                      f"Train Acc: {history['train_acc'][-1]:.4f}, "
                      f"Val Loss: {history['val_loss'][-1]:.4f}, "
                      f"Val Acc: {history['val_acc'][-1]:.4f}")

        print(f"✓ EEGNet training complete")
        print(f"  Best val accuracy: {max(history['val_acc']):.4f}")

        return model, history


    def save_pytorch_model(model, filepath: str):
        """
        Save PyTorch model

        Parameters
        ----------
        model : nn.Module
            PyTorch model
        filepath : str
            Path to save model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(model.state_dict(), filepath)
        print(f"✓ PyTorch model saved to {filepath}")


    def load_pytorch_model(model_class, filepath: str, **kwargs):
        """
        Load PyTorch model

        Parameters
        ----------
        model_class : class
            Model class (e.g., EEGNet)
        filepath : str
            Path to saved model
        **kwargs : dict
            Model initialization parameters

        Returns
        -------
        model : nn.Module
            Loaded model
        """
        model = model_class(**kwargs)
        model.load_state_dict(torch.load(filepath))
        model.eval()
        print(f"✓ PyTorch model loaded from {filepath}")
        return model


else:
    # Placeholder functions when PyTorch is not available
    def train_eegnet(*args, **kwargs):
        raise ImportError("PyTorch is not installed. Install with: pip install torch")


    def save_pytorch_model(*args, **kwargs):
        raise ImportError("PyTorch is not installed. Install with: pip install torch")


    def load_pytorch_model(*args, **kwargs):
        raise ImportError("PyTorch is not installed. Install with: pip install torch")