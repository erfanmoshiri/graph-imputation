"""
Evaluation metrics for binary feature imputation on benchmark datasets.
Implements Recall@K and NDCG@K metrics commonly used for binary data imputation.
"""

import torch
import numpy as np
from sklearn.metrics import ndcg_score


def recall_at_k(y_true, y_pred, mask, k=10):
    """
    Compute Recall@K for binary imputation.

    For each sample, rank predicted values and check if true positives
    are in the top-K predictions.

    Args:
        y_true: [num_nodes, num_features] - ground truth binary features
        y_pred: [num_nodes, num_features] - predicted continuous values
        mask: [num_nodes, num_features] - mask indicating which values to evaluate
        k: int - top-K threshold

    Returns:
        recall_k: float - average Recall@K across all masked samples
    """
    y_true = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_pred = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
    mask = mask if isinstance(mask, np.ndarray) else mask

    num_nodes = y_true.shape[0]
    recall_scores = []

    for i in range(num_nodes):
        # Get masked positions for this node
        node_mask = mask[i]

        if not node_mask.any():
            continue

        # Get true binary labels for masked positions
        true_labels = y_true[i, node_mask]

        # Get predicted scores for masked positions
        pred_scores = y_pred[i, node_mask]

        # Get indices of true positives (where true label = 1)
        true_positive_indices = np.where(true_labels == 1)[0]

        if len(true_positive_indices) == 0:
            # No positive labels for this node, skip
            continue

        # Rank predictions (descending order)
        ranked_indices = np.argsort(-pred_scores)

        # Get top-K predictions
        top_k_indices = ranked_indices[:k]

        # Count how many true positives are in top-K
        hits = len(set(true_positive_indices) & set(top_k_indices))

        # Recall = hits / total true positives
        recall = hits / len(true_positive_indices)
        recall_scores.append(recall)

    return np.mean(recall_scores) if recall_scores else 0.0


def ndcg_at_k(y_true, y_pred, mask, k=10):
    """
    Compute NDCG@K for binary imputation.

    Normalized Discounted Cumulative Gain measures ranking quality
    by giving higher weight to correct predictions at top positions.

    Args:
        y_true: [num_nodes, num_features] - ground truth binary features
        y_pred: [num_nodes, num_features] - predicted continuous values
        mask: [num_nodes, num_features] - mask indicating which values to evaluate
        k: int - top-K threshold

    Returns:
        ndcg_k: float - average NDCG@K across all masked samples
    """
    y_true = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_pred = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
    mask = mask if isinstance(mask, np.ndarray) else mask

    num_nodes = y_true.shape[0]
    ndcg_scores = []

    for i in range(num_nodes):
        # Get masked positions for this node
        node_mask = mask[i]

        if not node_mask.any():
            continue

        # Get true binary labels for masked positions
        true_labels = y_true[i, node_mask].reshape(1, -1)  # [1, num_masked]

        # Get predicted scores for masked positions
        pred_scores = y_pred[i, node_mask].reshape(1, -1)  # [1, num_masked]

        # Skip if all zeros (no positive labels)
        if true_labels.sum() == 0:
            continue

        # Compute NDCG@K using sklearn
        try:
            ndcg = ndcg_score(true_labels, pred_scores, k=k)
            ndcg_scores.append(ndcg)
        except:
            # Handle edge cases (e.g., all same predictions)
            continue

    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def evaluate_binary_imputation(y_true, y_pred, mask, k_values=[5, 10, 20]):
    """
    Comprehensive evaluation for binary imputation.

    Args:
        y_true: [num_nodes, num_features] - ground truth binary features
        y_pred: [num_nodes, num_features] - predicted continuous values
        mask: [num_nodes, num_features] - test mask
        k_values: list of K values for evaluation

    Returns:
        results: dict with Recall@K and NDCG@K for each K
    """
    results = {}

    for k in k_values:
        recall_k = recall_at_k(y_true, y_pred, mask, k=k)
        ndcg_k = ndcg_at_k(y_true, y_pred, mask, k=k)

        results[f'Recall@{k}'] = recall_k
        results[f'NDCG@{k}'] = ndcg_k

    return results


def print_binary_evaluation_results(results):
    """
    Pretty print evaluation results.

    Args:
        results: dict from evaluate_binary_imputation
    """
    print("\n" + "="*50)
    print("Binary Imputation Evaluation Results")
    print("="*50)

    # Group by metric type
    recall_metrics = {k: v for k, v in results.items() if 'Recall' in k}
    ndcg_metrics = {k: v for k, v in results.items() if 'NDCG' in k}

    if recall_metrics:
        print("\nRecall@K:")
        for metric, value in sorted(recall_metrics.items()):
            print(f"  {metric}: {value:.4f}")

    if ndcg_metrics:
        print("\nNDCG@K:")
        for metric, value in sorted(ndcg_metrics.items()):
            print(f"  {metric}: {value:.4f}")

    print("="*50 + "\n")
