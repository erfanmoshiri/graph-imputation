"""
Data loader for benchmark datasets (Cora, Citeseer, Pubmed, etc.)
Prepares data in format expected by the model:
- Numerical features followed by categorical features
- Creates imputation masks
- Normalizes features
"""

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, CitationFull
from sklearn.preprocessing import MinMaxScaler
import joblib


def load_cora(path='./data', mask_ratio=0.4):
    """
    Load Cora dataset for imputation task.

    Args:
        path: Path to store/load data
        mask_ratio: Ratio of features to mask for imputation

    Returns:
        data: PyTorch Geometric Data object with masks
    """
    dataset = Planetoid(root=path, name='Cora')
    original_data = dataset[0]

    print(f"Cora dataset loaded:")
    print(f"  Nodes: {original_data.num_nodes}")
    print(f"  Edges: {original_data.num_edges}")
    print(f"  Features: {original_data.num_features}")
    print(f"  Classes: {dataset.num_classes}")

    # All features are numerical (binary bag-of-words)
    features = original_data.x.numpy()

    # Normalize to [0, 1]
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features)

    # Create imputation masks
    num_nodes, num_features = features_normalized.shape
    total_values = num_nodes * num_features

    train_mask_size = int(0.8 * total_values * mask_ratio)
    test_mask_size = int(0.2 * total_values * mask_ratio)

    all_indices = np.arange(total_values)
    np.random.shuffle(all_indices)

    train_indices = all_indices[:train_mask_size]
    test_indices = all_indices[train_mask_size:train_mask_size + test_mask_size]

    train_mask = np.full((num_nodes, num_features), False)
    test_mask = np.full((num_nodes, num_features), False)

    train_row_indices, train_col_indices = np.unravel_index(train_indices, (num_nodes, num_features))
    test_row_indices, test_col_indices = np.unravel_index(test_indices, (num_nodes, num_features))

    train_mask[train_row_indices, train_col_indices] = True
    test_mask[test_row_indices, test_col_indices] = True

    # Create ground truth (original features)
    y = torch.tensor(features_normalized, dtype=torch.float)

    # Create input with masked values (set to 0 or use simple imputation)
    x = features_normalized.copy()
    x[train_mask] = 0  # Mask train values
    x[test_mask] = 0   # Mask test values

    # Simple mean imputation for masked values
    for j in range(num_features):
        col_mask = train_mask[:, j] | test_mask[:, j]
        if col_mask.sum() < num_nodes:
            mean_val = x[~col_mask, j].mean()
            x[col_mask, j] = mean_val

    x = torch.tensor(x, dtype=torch.float)

    # Create Data object
    data = Data(
        x=x,
        y=y,
        edge_index=original_data.edge_index,
        train_mask=train_mask,
        test_mask=test_mask,
        num_features=num_features,
        num_classes=num_features,  # For reconstruction
        df=None  # No DataFrame for benchmark datasets (uses single cluster mode)
    )

    return data


def load_citeseer(path='./data', mask_ratio=0.4):
    """
    Load Citeseer dataset for imputation task.

    Args:
        path: Path to store/load data
        mask_ratio: Ratio of features to mask for imputation

    Returns:
        data: PyTorch Geometric Data object with masks
    """
    dataset = Planetoid(root=path, name='Citeseer')
    original_data = dataset[0]

    print(f"Citeseer dataset loaded:")
    print(f"  Nodes: {original_data.num_nodes}")
    print(f"  Edges: {original_data.num_edges}")
    print(f"  Features: {original_data.num_features}")
    print(f"  Classes: {dataset.num_classes}")

    # All features are numerical (binary bag-of-words)
    features = original_data.x.numpy()

    # Normalize to [0, 1]
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features)

    # Create imputation masks
    num_nodes, num_features = features_normalized.shape
    total_values = num_nodes * num_features

    train_mask_size = int(0.8 * total_values * mask_ratio)
    test_mask_size = int(0.2 * total_values * mask_ratio)

    all_indices = np.arange(total_values)
    np.random.shuffle(all_indices)

    train_indices = all_indices[:train_mask_size]
    test_indices = all_indices[train_mask_size:train_mask_size + test_mask_size]

    train_mask = np.full((num_nodes, num_features), False)
    test_mask = np.full((num_nodes, num_features), False)

    train_row_indices, train_col_indices = np.unravel_index(train_indices, (num_nodes, num_features))
    test_row_indices, test_col_indices = np.unravel_index(test_indices, (num_nodes, num_features))

    train_mask[train_row_indices, train_col_indices] = True
    test_mask[test_row_indices, test_col_indices] = True

    # Create ground truth (original features)
    y = torch.tensor(features_normalized, dtype=torch.float)

    # Create input with masked values
    x = features_normalized.copy()
    x[train_mask] = 0
    x[test_mask] = 0

    # Simple mean imputation for masked values
    for j in range(num_features):
        col_mask = train_mask[:, j] | test_mask[:, j]
        if col_mask.sum() < num_nodes:
            mean_val = x[~col_mask, j].mean()
            x[col_mask, j] = mean_val

    x = torch.tensor(x, dtype=torch.float)

    # Create Data object
    data = Data(
        x=x,
        y=y,
        edge_index=original_data.edge_index,
        train_mask=train_mask,
        test_mask=test_mask,
        num_features=num_features,
        num_classes=num_features,
        df=None  # No DataFrame for benchmark datasets (uses single cluster mode)
    )

    return data


def load_benchmark_dataset(dataset_name, path='./data', mask_ratio=0.4):
    """
    Universal loader for benchmark datasets.

    Args:
        dataset_name: Name of dataset ('Cora', 'Citeseer', 'Pubmed')
        path: Path to store/load data
        mask_ratio: Ratio of features to mask

    Returns:
        data: PyTorch Geometric Data object
    """
    if dataset_name.lower() == 'cora':
        return load_cora(path, mask_ratio)
    elif dataset_name.lower() == 'citeseer':
        return load_citeseer(path, mask_ratio)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Use 'Cora' or 'Citeseer'.")
