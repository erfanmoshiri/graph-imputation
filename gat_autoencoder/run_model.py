import argparse
import os

from gat_autoencoder.data_preprocessing import load_data
from gat_autoencoder.train_test import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_decay_step', type=int, default=1000)
    parser.add_argument('--opt_decay_rate', type=float, default=0.9)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--log_dir', type=str, default='s5-01-m50')
    parser.add_argument('--opt_scheduler', type=str, default='step')

    parser.add_argument('--num_classes', type=int, default=10)  # indicated size of output layer of prediction
    parser.add_argument('--num_heads', type=int, default=3)

    # Feature configuration
    parser.add_argument('--num_features', type=int, default=10, help='Number of numerical features')
    parser.add_argument('--cat_feature_sizes', type=int, nargs='*', default=[5, 3, 4],
                        help='List of categorical feature sizes (e.g., 5 3 4 for 3 categorical features)')
    parser.add_argument('--latent_dim', type=int, default=10, help='Latent dimension size')

    # Attention-based neighborhood
    parser.add_argument('--neighbour_attention', action='store_true', default=False,
                        help='Enable attention-based neighborhood calculation')
    parser.add_argument('--gamma1', type=float, default=1.0, help='Weight for 1-hop attention')
    parser.add_argument('--gamma2', type=float, default=0.5, help='Weight for 2-hop attention')
    parser.add_argument('--gamma3', type=float, default=0.25, help='Weight for 3-hop attention')

    # Self-supervised learning
    parser.add_argument('--use_neighbour_residual', action='store_true', default=False,
                        help='Enable neighbour-residual SSL objective')
    parser.add_argument('--use_neighbour_diversity', action='store_true', default=False,
                        help='Enable neighbour-diversity SSL objective')
    parser.add_argument('--ssl_weight_residual', type=float, default=0.1,
                        help='Weight for neighbour-residual SSL loss')
    parser.add_argument('--ssl_weight_diversity', type=float, default=0.1,
                        help='Weight for neighbour-diversity SSL loss')
    parser.add_argument('--k_neighbours', type=int, default=5, help='Number of top neighbors to keep in A_eff')

    # Binary evaluation for benchmark datasets
    parser.add_argument('--binary_eval', action='store_true', default=False,
                        help='Use binary evaluation metrics (Recall@K, NDCG@K) for benchmark datasets')
    parser.add_argument('--eval_k_values', type=int, nargs='+', default=[5, 10, 20],
                        help='K values for Recall@K and NDCG@K evaluation')

    subparsers = parser.add_subparsers()
    args = parser.parse_args()
    args.is_eval = True

    path = './' + args.log_dir + '/'
    os.makedirs(path, exist_ok=args.is_eval)
    data = load_data(path, args)
    run(data, path, args)

