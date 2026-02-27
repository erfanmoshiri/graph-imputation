import json
import os
import os.path as osp

import joblib
import numpy as np
import pandas as pd
import torch
from joblib import dump
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch_geometric.data import Data


# data_df = pd.read_hdf('/GAT_clean_df_sorted.hdf', key='s')


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


def create_edge_index(df):
    size = df.shape[0]
    edge_start = list(range(size - 1))
    edge_end = list(range(1, size))
    edge_end2 = list(range(2, size))
    edge_end2.extend(([size - 1]))
    edge_end.extend(edge_end2)
    return torch.tensor([edge_start * 2, edge_end], dtype=int)


def get_train_test_mask(df, path, mask_level, from_file=False):
    if from_file:
        mask_dict = joblib.load(path + 'mask_40p.pkl')
        return mask_dict['train_mask'], mask_dict['test_mask']

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

    if not from_file:
        joblib.dump({'train_mask': train_mask, 'test_mask': test_mask}, path + 'mask_40p.pkl')

    return train_mask, test_mask


def mask_categorical_values_by_category(df, mask_prob=0.2, train_ratio=0.8):
    train_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    test_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    # Split the rows for train and test
    num_rows = len(df)
    train_size = int(train_ratio * num_rows)
    test_size = num_rows - train_size

    # Shuffle the data indices to randomize the split
    indices = np.random.permutation(num_rows)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Create mask function
    def create_mask_for_category(category_range, mask_prob):
        mask = np.random.rand(len(df)) < mask_prob
        category_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        category_mask.iloc[:, category_range] = np.repeat(mask[:, np.newaxis], len(category_range), axis=1)
        return category_mask

    # Generate train mask
    train_category1_mask = create_mask_for_category(range(10, 15), mask_prob)
    train_category2_mask = create_mask_for_category(range(15, 18), mask_prob)
    train_category3_mask = create_mask_for_category(range(18, 22), mask_prob)

    # Apply train mask to the training indices only
    train_mask.iloc[train_indices, 10:15] = train_category1_mask.iloc[train_indices, 10:15]
    train_mask.iloc[train_indices, 15:18] = train_category2_mask.iloc[train_indices, 15:18]
    train_mask.iloc[train_indices, 18:22] = train_category3_mask.iloc[train_indices, 18:22]

    # Generate test mask
    test_category1_mask = create_mask_for_category(range(10, 15), mask_prob)
    test_category2_mask = create_mask_for_category(range(15, 18), mask_prob)
    test_category3_mask = create_mask_for_category(range(18, 22), mask_prob)

    # Apply test mask to the test indices only (ensuring no overlap with train mask)
    test_mask.iloc[test_indices, 10:15] = test_category1_mask.iloc[test_indices, 10:15]
    test_mask.iloc[test_indices, 15:18] = test_category2_mask.iloc[test_indices, 15:18]
    test_mask.iloc[test_indices, 18:22] = test_category3_mask.iloc[test_indices, 18:22]

    return train_mask.iloc[:, 10:].values, test_mask.iloc[:, 10:].values


def get_data(df, args, path):
    edge_index = create_edge_index(df)

    col_names = ['POSTCODE', 'SUBURB_NAME', 'STREET_NAME', 'STREET_TYPE', 'STREET_NUMBER']
    df_clean = df.drop(columns=col_names)

    train_mask, test_mask = get_train_test_mask(df_clean, path, 0.4, from_file=args.is_eval)
    cat_train_mask, cat_test_mask = mask_categorical_values_by_category(df_clean, mask_prob=0.4)
    df.fillna(0, inplace=True)

    x = df_clean.values.astype(np.float64)
    y = torch.tensor(df_clean.values.astype(np.float64), dtype=torch.float)

    x[:, :10][train_mask] = np.nan
    x[:, :10][test_mask] = np.nan
    x[:, 10:][cat_train_mask] = 0
    x[:, 10:][cat_test_mask] = 0

    imputer = SimpleImputer(strategy='mean')

    # Apply the imputer to the selected numerical columns
    x = imputer.fit_transform(x)
    x = torch.tensor(x.astype(np.float64), dtype=torch.float)

    new_df = pd.DataFrame(x, columns=df_clean.columns)
    t = df[col_names].copy().reset_index(drop=True)
    new_df_concat = pd.concat([t, new_df.reset_index(drop=True)], axis=1)

    data = Data(x=x, y=y,
                edge_index=edge_index,
                train_mask=train_mask, test_mask=test_mask,
                cat_train_mask=cat_train_mask, cat_test_mask=cat_test_mask,
                num_features=x.shape[1],
                num_classes=10,
                df=new_df_concat,
                df_y=df
                )

    return data


def write_features_to_file(df, path):
    os.makedirs(path, exist_ok=True)

    with open(osp.join(path, 'data_features.txt'), 'w') as file:
        for column_name in df.columns:
            file.write(column_name + '\n')


def find_weight_for_cats():
    df = pd.read_hdf('../data/new_log_minmax.x', key='s')
    df = df[df['ROOF_SHAPE_OTHER'] == 0]
    df = df.drop(columns=['ROOF_SHAPE_OTHER'])
    df = df.iloc[:40000]

    roof_material_columns = [
        'PRIMARY_ROOF_MATERIAL_-1', 'PRIMARY_ROOF_MATERIAL_FIBERGLASS/PLASTIC',
        'PRIMARY_ROOF_MATERIAL_FLAT CONCRETE', 'PRIMARY_ROOF_MATERIAL_METAL',
        'PRIMARY_ROOF_MATERIAL_TILE'
    ]

    roof_shape_columns = [
        'ROOF_SHAPE_-1', 'ROOF_SHAPE_FLAT', 'ROOF_SHAPE_GABLED', 'ROOF_SHAPE_HIPPED',
        'ROOF_SHAPE_MANSARD', 'ROOF_SHAPE_MIXED', 'ROOF_SHAPE_SHED'
    ]

    property_type_columns = [
        'CL_PROPERTY_TYPE_CATEGORY_Apartment/Unit', 'CL_PROPERTY_TYPE_CATEGORY_House',
        'CL_PROPERTY_TYPE_CATEGORY_Rural/Farming', 'CL_PROPERTY_TYPE_CATEGORY_Townhouse'
    ]

    cat1_counts = df[roof_material_columns].sum()
    cat2_counts = df[roof_shape_columns].sum()
    cat3_counts = df[property_type_columns].sum()

    def calculate_class_weights(one_hot_counts):
        total_samples = one_hot_counts.sum()
        class_weights = total_samples / (len(one_hot_counts) * one_hot_counts)
        return class_weights / 2

    # Calculate weights for each category
    cat1_weights = calculate_class_weights(cat1_counts)
    cat2_weights = calculate_class_weights(cat2_counts)
    cat3_weights = calculate_class_weights(cat3_counts)

    return cat1_weights, cat2_weights, cat3_weights


def load_data(path, args):
    # data_obj = torch.load('./uci/test/data_10p.pt')
    # print(data_obj)
    # return data_obj

    df = pd.read_hdf('../data/log_minmax_clean_df_sorted_4.hdf', key='s')
    df = df[(df['POSTCODE'] >= '2000') & (df['POSTCODE'] < '3100')]
    df = df.drop(columns=['ESTIMATED_LEVELS'])
    print(df.shape[0])
    # df = df[df['ROOF_SHAPE_OTHER'] == 0]
    # df = df.drop(columns=['ROOF_SHAPE_OTHER'])
    # df = df.iloc[:40000]
    # df = sample_onehot(df)
    # df = df.iloc[:26000]
    # df = pd.read_hdf('../data/new_sampled.x', key='s')

    # df = reduce_feat(df)
    # df = sample_per_category(df)
    # df = under_sample_df(df, 400)

    print('df shape: ', df.shape)
    write_features_to_file(df, path)
    data = get_data(df, args, path)
    return data
