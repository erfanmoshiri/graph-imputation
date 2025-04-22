import json
import os
import os.path as osp

import joblib
import numpy as np
import pandas as pd
import torch
from joblib import dump
from sklearn.impute import SimpleImputer
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
