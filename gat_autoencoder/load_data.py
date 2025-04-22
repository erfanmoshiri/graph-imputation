import json
import os
import os.path as osp

import joblib
import numpy as np
import pandas as pd
import torch
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def replace_categorical_null(arr):
    for i in range(arr.shape[1]):
        col = arr[:, i]
        count_zeros = np.sum(col == 0)
        count_ones = np.sum(col == 1)

        most_common = 1 if count_ones >= count_zeros else 0
        arr[np.isnan(arr[:, i]), i] = most_common
    return arr


def evaluate_knn_imputation(full_data, missing_data, imputer='mean', n_neighbors=3):
    print('starting knn imputation for evaluation')

    mask = np.isnan(missing_data)

    if imputer == 'knn':
        imputer_model = KNNImputer(n_neighbors=n_neighbors)
    else:
        imputer_model = SimpleImputer(strategy='mean')

    imputed_data = imputer_model.fit_transform(missing_data)

    full_imputed_values = full_data[mask]
    imputed_values = imputed_data[mask]

    mse = mean_squared_error(full_imputed_values, imputed_values)
    mae = mean_absolute_error(full_imputed_values, imputed_values)
    r2 = r2_score(full_imputed_values, imputed_values)

    print('done eval')
    return mse, mae, r2


def cluster_knn_imputer(df, df_y, n_neighbors=2):
    # Initialize total metrics
    total_mse, total_mae, total_r2 = 0, 0, 0
    cluster_count = 0

    # Group the DataFrame based on specified columns
    groups = df.groupby(['POSTCODE', 'SUBURB_NAME'])
    groups_y = df_y.groupby(['POSTCODE', 'SUBURB_NAME'])

    # Process each group
    for (key, group), (key_Y, group_y) in zip(groups, groups_y):
        if len(group) <= 2:
            continue
        # Reset index for easier handling
        group = group.reset_index(drop=True)
        group_y = group_y.reset_index(drop=True)

        arr = group.values[:, 5:15]
        arr_y = group_y.values[:, 5:15]

        # Create a mask of missing values (True for missing, False for present)
        arr = np.array(arr, dtype='float')
        arr_y = np.array(arr_y, dtype='float')
        mask = np.isnan(arr)

        # Apply KNN Imputer
        imputer = KNNImputer(n_neighbors=n_neighbors)

        # Impute missing values using KNNImputer
        imputed_data = imputer.fit_transform(arr)

        # Get only the imputed values using the mask
        if mask.shape[1] != imputed_data.shape[1]:
            continue

        imputed_values = imputed_data[mask]
        true_values = arr_y[mask]

        if np.isnan(imputed_values).any():
            print("Some rows have only NaNs, skipping them")
            continue  # Skip this group or handle differently

        if len(imputed_values) > 0:
            mse = mean_squared_error(true_values, imputed_values)
            mae = mean_absolute_error(true_values, imputed_values)
            r2 = r2_score(true_values, imputed_values)

            if mse is None or mae is None or r2 is None:
                print('sth is none')
            # Accumulate metrics
            total_mse += mse
            total_mae += mae
            total_r2 += r2
            cluster_count += 1

    # Calculate the average MSE, MAE, and R2 across all clusters
    avg_mse = total_mse / cluster_count if cluster_count > 0 else None
    avg_mae = total_mae / cluster_count if cluster_count > 0 else None
    avg_r2 = total_r2 / cluster_count if cluster_count > 0 else None

    return avg_mse, avg_mae, avg_r2


def knn_imputer(arr):
    missing_mask = np.isnan(arr)

    imputer = SimpleImputer(strategy='mean')
    arr_imputed = imputer.fit_transform(arr)

    nn = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(arr_imputed)
    distances, indices = nn.kneighbors(arr_imputed)

    for i in range(arr_imputed.shape[0]):
        nearest_neighbor_idx = indices[i, 1]

        for j in range(arr_imputed.shape[1]):
            if missing_mask[i, j]:
                arr_imputed[i, j] = arr_imputed[nearest_neighbor_idx, j]

    return arr_imputed


def get_train_test_mask(df, path, mask_level, from_file=False):
    if from_file:
        mask_dict = joblib.load(path + 'mask_40p.pkl')
        return mask_dict['train_mask'], mask_dict['test_mask'], mask_dict['cat_mask'].iloc[:, 10:].values

    df_choose = df.iloc[:, :10]
    total_values = df_choose.size

    train_values = int(0.8 * total_values * mask_level)
    test_values = int(0.2 * total_values * mask_level)

    all_indices = np.arange(total_values)
    np.random.shuffle(all_indices)
    train_indices = all_indices[:train_values]
    test_indices = all_indices[train_values: train_values + test_values]

    train_mask = np.full(df_choose.shape, False)
    test_mask = np.full(df_choose.shape, False)

    train_row_indices, train_col_indices = np.unravel_index(train_indices, df_choose.shape)
    test_row_indices, test_col_indices = np.unravel_index(test_indices, df_choose.shape)

    train_mask[train_row_indices, train_col_indices] = True
    test_mask[test_row_indices, test_col_indices] = True

    # MASKING CATEGORICAL FEATURES
    cat_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    category1_mask = np.random.rand(len(df)) < mask_level
    cat_mask.iloc[:, 10:15] = np.repeat(category1_mask[:, np.newaxis], 5, axis=1)

    category2_mask = np.random.rand(len(df)) < mask_level
    cat_mask.iloc[:, 15:18] = np.repeat(category2_mask[:, np.newaxis], 3, axis=1)

    category3_mask = np.random.rand(len(df)) < mask_level
    cat_mask.iloc[:, 18:22] = np.repeat(category3_mask[:, np.newaxis], 4, axis=1)

    if not from_file:
        joblib.dump({'train_mask': train_mask, 'test_mask': test_mask, 'cat_mask': cat_mask}, path + 'mask_40p.pkl')

    return train_mask, test_mask, cat_mask.iloc[:, 10:].values


def get_data(df, args, path):
    # train_indices, test_indices = get_mask(df['Y'].to_numpy())
    col_names = ['POSTCODE', 'SUBURB_NAME', 'STREET_NAME', 'STREET_TYPE', 'STREET_NUMBER']
    df_clean = df.drop(columns=col_names)

    train_mask, test_mask, cat_mask = get_train_test_mask(df_clean, path, 0.4, from_file=args.is_eval)
    # df.fillna(0, inplace=True)

    x = df_clean.values.astype(np.float64)
    y = torch.tensor(df_clean.values.astype(np.float64), dtype=torch.float)

    x[:, :10][train_mask] = np.nan
    x[:, :10][test_mask] = np.nan
    x[:, 10:][cat_mask] = 0

    # mse, mae, r2 = evaluate_knn_imputation(df_clean.values, x)
    # print(mse, mae, r2)

    x[:, 10:] = replace_categorical_null(x[:, 10:])
    print('starting initial knn imputation')
    x[:, :10] = knn_imputer(x[:, :10])

    # re = knn_imputer(x[:, :10])
    # y_for_eval = df_clean.values.astype(np.float64)[:, :10]
    # mask = train_mask | test_mask
    # mse = mean_squared_error(y_for_eval[mask], re[mask])
    # mae = mean_absolute_error(y_for_eval[mask], re[mask])
    # r2 = r2_score(y_for_eval[mask], re[mask])
    # print(mse, mae, r2)

    print('finished initial knn imputation')

    x = torch.tensor(x.astype(np.float64), dtype=torch.float)

    new_df = pd.DataFrame(x, columns=df_clean.columns)
    t = df[col_names].copy().reset_index(drop=True)
    new_df_concat = pd.concat([t, new_df.reset_index(drop=True)], axis=1)

    data = Data(x=x, y=y,
                train_mask=train_mask, test_mask=test_mask,
                cat_mask=cat_mask,
                num_features=x.shape[1],
                num_classes=10,
                df=new_df_concat,
                df_y=df
                )

    print('VALIDATING:', new_df_concat.iloc[34, 14], df.iloc[34, 14])

    return data


def write_features_to_file(df, path):
    os.makedirs(path, exist_ok=True)

    with open(osp.join(path, 'data_features.txt'), 'w') as file:
        for column_name in df.columns:
            file.write(column_name + '\n')


def load_data(path, args):
    # data_obj = torch.load('./uci/test/data_10p.pt')
    # print(data_obj)
    # return data_obj

    # df = pd.read_hdf('./data/new_log_minmax.x', key='s')
    # df = sample_onehot(df)
    # df = df.iloc[:26000]
    # df = pd.read_hdf('./data/new_sampled.x', key='s')

    df = pd.read_hdf('data/log_minmax_clean_df_sorted_4.hdf', key='s')
    df = df[(df['POSTCODE'] >= '2000') & (df['POSTCODE'] < '3100')]
    df = df.drop(columns=['ESTIMATED_LEVELS'])
    # df = reduce_feat(df)
    # df = sample_per_category(df)
    # df = under_sample_df(df, 400)

    print('df shape: ', df.shape)
    write_features_to_file(df, path)
    data = get_data(df, args, path)
    return data
