import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data_equally(df):
    # Assuming df is your DataFrame
    grouped = df.groupby('Y', group_keys=False)

    train_indices = []
    test_indices = []

    # Iterate over each group
    for group_name, group_indices in grouped.indices.items():
        if group_name == 10 and len(group_indices) > 500000:
            continue
        # Split the group indices into train and test sets
        group_train_indices, group_test_indices = train_test_split(group_indices.astype(int).tolist(), test_size=0.3)
        print(len(group_train_indices), len(group_test_indices))
        # Append the train and test indices to lists
        train_indices.extend(group_train_indices)
        test_indices.extend(group_test_indices)
    return train_indices, test_indices


if __name__ == '__main__':
    root_path = './data/data_split_2'
    df = pd.read_hdf('./data/GAT_clean_df_sorted.hdf', key='s')
    os.makedirs(root_path)

    for i in range(5):
        train_indices, test_indices = split_data_equally(df)
        indices_dict = {'train_indices': train_indices, 'test_indices': test_indices}
        path = f'{root_path}/{i+1}'
        os.makedirs(path)

        with open(os.path.join(path, 'indices.xyz'), 'w') as f:
            json.dump(indices_dict, f)

    df.to_hdf(f'{root_path}/df_main_sorted.hdf', key='s')
    print('write successful')
