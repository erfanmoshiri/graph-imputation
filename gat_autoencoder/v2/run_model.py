import argparse
import os

from gat_autoencoder.v2.load_data import load_data
from gat_autoencoder.v2.train_model import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_decay_step', type=int, default=1000)
    parser.add_argument('--opt_decay_rate', type=float, default=0.9)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--log_dir', type=str, default='r5-06')
    parser.add_argument('--opt_scheduler', type=str, default='step')
    parser.add_argument('--noise_std', type=float, default='0.01')

    parser.add_argument('--num_classes', type=int, default=8)  # indicated size of output layer of prediction
    parser.add_argument('--num_heads', type=int, default=3)

    subparsers = parser.add_subparsers()
    args = parser.parse_args()
    args.is_eval = False

    path = './' + args.log_dir + '/'
    os.makedirs(path, exist_ok=args.is_eval)
    data = load_data(path, args)
    run(data, path, args)

