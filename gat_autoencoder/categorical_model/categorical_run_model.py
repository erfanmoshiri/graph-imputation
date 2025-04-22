import argparse
import os

from gat_autoencoder.categorical_model.categorical_load_data import load_data
from gat_autoencoder.categorical_model.categorical_train import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_decay_step', type=int, default=1000)
    parser.add_argument('--opt_decay_rate', type=float, default=0.9)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--log_dir', type=str, default='s04-02')
    parser.add_argument('--opt_scheduler', type=str, default='step')

    parser.add_argument('--num_classes', type=int, default=10)  # indicated size of output layer of prediction
    parser.add_argument('--num_heads', type=int, default=3)

    subparsers = parser.add_subparsers()
    args = parser.parse_args()
    args.is_eval = True

    path = './' + args.log_dir + '/'
    os.makedirs(path, exist_ok=args.is_eval)
    data = load_data(path, args)
    run(data, path, args)

