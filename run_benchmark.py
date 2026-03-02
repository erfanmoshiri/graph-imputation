"""
Runner script for benchmark datasets (Cora, Citeseer).
"""

import argparse
import os
from gat_autoencoder.benchmark_data_loader import load_benchmark_dataset
from gat_autoencoder.train_test import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model on benchmark datasets')

    # Dataset selection
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'Citeseer'],
                        help='Benchmark dataset to use')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path to store/load dataset')
    parser.add_argument('--mask_ratio', type=float, default=0.4,
                        help='Ratio of features to mask for imputation')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_decay_step', type=int, default=1000)
    parser.add_argument('--opt_decay_rate', type=float, default=0.9)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--log_dir', type=str, default='benchmark_results')
    parser.add_argument('--opt_scheduler', type=str, default='step')

    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Latent dimension size')
    parser.add_argument('--num_heads', type=int, default=3)

    # Attention-based neighborhood
    parser.add_argument('--neighbour_attention', action='store_true', default=False)
    parser.add_argument('--gamma1', type=float, default=1.0)
    parser.add_argument('--gamma2', type=float, default=0.5)
    parser.add_argument('--gamma3', type=float, default=0.25)

    # Self-supervised learning
    parser.add_argument('--use_neighbour_residual', action='store_true', default=False)
    parser.add_argument('--use_neighbour_diversity', action='store_true', default=False)
    parser.add_argument('--ssl_weight_residual', type=float, default=0.1)
    parser.add_argument('--ssl_weight_diversity', type=float, default=0.1)
    parser.add_argument('--k_neighbours', type=int, default=5)

    # Binary evaluation
    parser.add_argument('--binary_eval', action='store_true', default=True,
                        help='Enable binary evaluation (Recall@K, NDCG@K)')
    parser.add_argument('--eval_k_values', type=int, nargs='+', default=[5, 10, 20],
                        help='K values for evaluation')

    args = parser.parse_args()
    args.is_eval = False  # Always train, then evaluate

    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    data = load_benchmark_dataset(args.dataset, path=args.data_path, mask_ratio=args.mask_ratio)

    # Set feature configuration based on dataset
    args.num_features = data.num_features
    args.cat_feature_sizes = []  # No categorical features for benchmark datasets
    args.num_classes = data.num_features

    print(f"\nDataset configuration:")
    print(f"  Nodes: {data.x.shape[0]}")
    print(f"  Features: {args.num_features}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Edges: {data.edge_index.shape[1]}")
    print(f"  Masked values: {data.test_mask.sum()}")

    # Create output directory
    output_path = f'./{args.log_dir}/{args.dataset.lower()}/'
    os.makedirs(output_path, exist_ok=True)

    # Run training and evaluation
    print(f"\nStarting training for {args.epochs} epochs...")
    run(data, output_path, args)

    print(f"\nResults saved to: {output_path}")
