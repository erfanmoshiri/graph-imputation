import json
import os
import os.path as osp

import joblib
import numpy as np
import pandas as pd
import torch
from joblib import dump
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data


# data_df = pd.read_hdf('/GAT_clean_df_sorted.hdf', key='s')


def create_edge_index(df):
    size = df.shape[0]
    edge_start = list(range(size - 1))
    edge_end = list(range(1, size))
    edge_end2 = list(range(2, size))
    edge_end2.extend(([size - 1]))
    edge_end.extend(edge_end2)
    return torch.tensor([edge_start * 2, edge_end], dtype=int)


def get_train_test_mask(df, path, mask_size, from_file=False):
    if from_file:
        mask_dict = joblib.load(path + 'mask_' + str(mask_size) + '.pkl')
        return mask_dict['train_mask'], mask_dict['test_mask']

    # else:
    #     mask_dict = joblib.load('mask_' + str(mask_size) + '.pkl')
    #     joblib.dump({'train_mask': mask_dict['train_mask'], 'test_mask': mask_dict['test_mask']}, path + 'mask_' + str(mask_size) + '.pkl')
    #     return mask_dict['train_mask'], mask_dict['test_mask']

    df_choose = df.iloc[:, :9]
    total_values = df_choose.size

    train_values = int(0.08 * total_values * mask_size)
    test_values = int(0.02 * total_values * mask_size)

    all_indices = np.arange(total_values)
    np.random.shuffle(all_indices)
    train_indices = all_indices[:train_values]
    test_indices = all_indices[train_values: train_values + test_values]

    train_mask = np.full(df_choose.shape, False)
    test_mask = np.full(df_choose.shape, False)
    # empty_rest_mask = np.full(df_false_shape, False)

    train_row_indices, train_col_indices = np.unravel_index(train_indices, df_choose.shape)
    test_row_indices, test_col_indices = np.unravel_index(test_indices, df_choose.shape)

    train_mask[train_row_indices, train_col_indices] = True
    test_mask[test_row_indices, test_col_indices] = True

    # train_mask_complete = np.concatenate((train_mask, empty_rest_mask), axis=1)
    # test_mask_complete = np.concatenate((test_mask, empty_rest_mask), axis=1)

    if not from_file:
        joblib.dump({'train_mask': train_mask, 'test_mask': test_mask}, path + 'mask_' + str(mask_size) + '.pkl')

    return train_mask, test_mask


def get_mask(arr):
    all_gt_mask = ~np.isnan(arr)

    test_count = int(np.sum(all_gt_mask) * 0.3)

    true_indices = np.where(all_gt_mask)[0]

    test_indices = np.random.choice(true_indices, size=test_count, replace=False)
    train_indices = np.setdiff1d(true_indices, test_indices)

    return train_indices, test_indices


def get_data(df, args, path):
    edge_index = create_edge_index(df)

    train_mask, test_mask = get_train_test_mask(df, path, 4, from_file=args.is_eval)

    #     df.fillna(0, inplace=True)
    #
    #     x = torch.tensor(df.values.astype(np.float64), dtype=torch.float)
    #     y = x.clone()
    #
    #     x[:, :9][train_mask] = np.nan
    #     x[:, :9][test_mask] = np.nan
    # #####################################################################
    x_np = df.values.astype(np.float64)

    x = torch.tensor(df.values.astype(np.float64), dtype=torch.float)
    y = x.clone()

    x_np[:, :9][train_mask] = np.nan  # Mask the specific cells for train
    x_np[:, :9][test_mask] = np.nan  # Mask the specific cells for test

    imputer = SimpleImputer(strategy='mean')
    x_np[:, :9] = imputer.fit_transform(x_np[:, :9])

    x = torch.tensor(x_np, dtype=torch.float)

    data = Data(x=x, y=y,
                edge_index=edge_index,
                train_mask=train_mask, test_mask=test_mask,
                num_features=x.shape[1],
                num_classes=args.num_classes
                )

    return data


def write_features_to_file(df, path):
    os.makedirs(path, exist_ok=True)

    with open(osp.join(path, 'data_features.txt'), 'w') as file:
        for column_name in df.columns:
            file.write(column_name + '\n')


def split_into_clusters(tensor, num_clusters):
    size = tensor.shape[0]
    cluster_size = size // num_clusters
    clusters = []

    for i in range(num_clusters):
        start_idx = i * cluster_size
        if i == num_clusters - 1:  # Last cluster takes remaining nodes
            end_idx = size
        else:
            end_idx = (i + 1) * cluster_size

        clusters.append(tensor[start_idx:end_idx])

    return clusters


def create_cluster_edge_index(cluster_size):
    # Create edges for a cluster (similar to your original edge creation)
    edge_start = list(range(cluster_size - 1))
    edge_end = list(range(1, cluster_size))
    edge_end2 = list(range(2, cluster_size))
    edge_end2.extend([cluster_size - 1])
    edge_end.extend(edge_end2)
    edge_index = torch.tensor([edge_start * 2, edge_end], dtype=torch.long)

    return edge_index


def add_gaussian_noise(input_tensor, mean=0.0, std=0.1):
    noise = torch.randn_like(input_tensor) * std + mean
    noisy_input = input_tensor + noise
    return torch.clamp(noisy_input, 0.0, 1.0)  # Clamping to keep values in [0, 1] range


def load_data(path, args):
    # df = pd.read_hdf('../data/new/processed_data_small.x', key='s')
    df = pd.read_hdf('../data/new/processed_data_small.x', key='s')
    df = df.iloc[:-2]
    # df = reduce_feat(df)
    # df = sample_per_category(df)
    # df = under_sample_df(df, 400)

    print('df shape: ', df.shape)
    write_features_to_file(df, path)
    data = get_data(df, args, path)
    return data
