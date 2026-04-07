import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class DLWFDataset(Dataset):
    def __init__(self, X, y):
        """
        Args:
            X: numpy array of shape (N, T, D)
            y: numpy array of shape (N,)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --------------------------------------------------
# NPZ loader
# --------------------------------------------------

def load_dlwf_npz(npz_path, normalize=False, label_encoder=None):
    """
    Load DLWF dataset from an .npz file.

    Args:
        npz_path (str): path to .npz file
        normalize (bool): whether to standardize X
        label_encoder (LabelEncoder or None):
            If provided, use this encoder to transform labels.
            If None, fit a new encoder.

    Returns:
        X (np.ndarray): shape (N, T, D)
        y (np.ndarray): shape (N,)
        le (LabelEncoder): fitted label encoder
    """
    data_obj = np.load(npz_path, allow_pickle=True)

    if "data" not in data_obj.files or "labels" not in data_obj.files:
        raise ValueError(
            f"Expected keys ['data', 'labels'], got {data_obj.files}"
        )

    X = data_obj["data"]      # expected shape (N, L)
    y_raw = data_obj["labels"]

    if X.ndim != 2:
        raise ValueError(f"Expected shape (N, L), got {X.shape}")

    # label encoding
    if label_encoder is None:
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
    else:
        le = label_encoder
        y = le.transform(y_raw)

    # convert to TS2Vec input shape: (N, T, D)
    X = X.astype(np.float32)
    X = X[..., np.newaxis]

    # optional normalization
    if normalize:
        mean = X.mean()
        std = X.std()
        if std > 0:
            X = (X - mean) / std

    return X, y, le


# --------------------------------------------------
# Build dataloaders
# --------------------------------------------------

def build_dlwf_dataloaders(
    npz_path,
    batch_size=32,
    normalize=False,
    val_size=0.1,
    test_size=0.1,
    random_state=42,
    num_workers=0
):
    """
    Build train/val/test DataLoaders from one DLWF npz file.

    Args:
        npz_path (str): path to .npz file
        batch_size (int): dataloader batch size
        normalize (bool): whether to normalize X
        val_size (float): validation ratio
        test_size (float): test ratio
        random_state (int): random seed
        num_workers (int): number of workers for DataLoader

    Returns:
        train_loader, val_loader, test_loader, label_encoder
    """
    X, y, le = load_dlwf_npz(npz_path, normalize=normalize)

    # first split out test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # then split train and val
    val_ratio_adjusted = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=y_train_val
    )

    train_dataset = DLWFDataset(X_train, y_train)
    val_dataset = DLWFDataset(X_val, y_val)
    test_dataset = DLWFDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, le