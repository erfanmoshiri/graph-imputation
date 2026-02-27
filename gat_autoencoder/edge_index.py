import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors


def create_edge_index_for_full_data(df):
    df = df.reset_index(drop=True)
    # Get total number of nodes in the dataset
    size = df.shape[0]

    # Generate neighborhood edges (previous and next, and two before/after) for all nodes
    edge_start = list(range(size - 1)) + list(range(size - 2))
    edge_end = list(range(1, size)) + list(range(2, size))

    # Add bidirectional edges (both directions)
    edge_indices = list(zip(edge_start, edge_end))
    edge_indices += list(zip(edge_end, edge_start))

    # Fully connected edges within groups by STREET_NAME
    street_groups = df.groupby(['POSTCODE', 'SUBURB_NAME', 'STREET_NAME', 'STREET_TYPE', 'STREET_NUMBER'])
    for _, street_group in street_groups:
        street_node_indices = street_group.index.values
        for i in range(len(street_node_indices)):
            for j in range(i + 1, len(street_node_indices)):
                # Connect all nodes within the same STREET_NAME group
                edge_indices.append((street_node_indices[i], street_node_indices[j]))
                edge_indices.append((street_node_indices[j], street_node_indices[i]))

                a = street_node_indices[i]
                b = street_node_indices[j]
                if a >= len(df) or b >= len(df):
                    print(a, b)

    edge_index = torch.tensor(list(zip(*edge_indices)), dtype=torch.long)
    return edge_index


def create_cluster_edge_index(cluster):
    # Get local size of the cluster
    size = len(cluster)
    if size == 1:
        return torch.tensor([[0], [0]], dtype=torch.long)

    # Generate neighborhood edges (previous and next, and two before/after)
    edge_start = list(range(size - 1)) + list(range(size - 2))
    edge_end = list(range(1, size)) + list(range(2, size))

    # Add bidirectional edges (both directions)
    edge_indices = list(zip(edge_start, edge_end))
    edge_indices += list(zip(edge_end, edge_start))

    # Fully connected edges within groups by STREET_NAME
    street_groups = cluster.groupby(['POSTCODE', 'SUBURB_NAME', 'STREET_NAME', 'STREET_TYPE'])
    for _, street_group in street_groups:
        street_node_indices = street_group.index.values  # Get the actual node indices for this group
        for i in range(len(street_node_indices)):
            for j in range(i + 1, len(street_node_indices)):
                # Connect all nodes within the same STREET_NAME group using the original indices
                edge_indices.append((street_node_indices[i], street_node_indices[j]))
                edge_indices.append((street_node_indices[j], street_node_indices[i]))  # Undirected edge

    n_neighbors = min(3, size)
    features = cluster.iloc[:, 5:15].values
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(features)
    distances, indices = nbrs.kneighbors(features)

    total_mae, total_mse, total_r2 = 0, 0, 0

    for i in range(size):
        # Add edges to the 2 nearest neighbors (excluding itself)
        for j in range(1, n_neighbors):  # Skip the first neighbor (itself)
            nearest_neighbor = indices[i, j]
            edge_indices.append((i, nearest_neighbor))
            edge_indices.append((nearest_neighbor, i))  # Bidirectional edge

    edge_index = torch.tensor(list(zip(*edge_indices)), dtype=torch.long)

    return edge_index


def create_clusters(df, df_y):
    # Generate clusters based on POSTCODE
    print('generating edges and clusters')
    clusters = df.groupby('POSTCODE')
    clusters_y = df_y.groupby('POSTCODE')
    assert len(clusters_y) == len(clusters)

    maes, mses, r2s = 0, 0, 0
    cluster_data = []
    for (key, cluster), (key_Y, cluster_y) in zip(clusters, clusters_y):
        assert len(cluster) == len(cluster_y)
        if len(cluster) == 1:
            continue
        # Reset the index of the cluster to create local indexing
        cluster = cluster.reset_index(drop=True)
        cluster_y = cluster_y.reset_index(drop=True)

        # Create edge index for the current cluster
        edge_index = create_cluster_edge_index(cluster)

        cluster = cluster.drop(columns=['POSTCODE', 'SUBURB_NAME', 'STREET_NAME', 'STREET_TYPE', 'STREET_NUMBER'])
        cluster_y = cluster_y.drop(columns=['POSTCODE', 'SUBURB_NAME', 'STREET_NAME', 'STREET_TYPE', 'STREET_NUMBER'])
        cluster = cluster.astype(float)
        cluster_y = cluster_y.astype(float)

        cluster = torch.tensor(cluster.values, dtype=torch.float)
        cluster_y = torch.tensor(cluster_y.values, dtype=torch.float)
        # Append the cluster data and edge index for use later
        cluster_data.append((cluster, cluster_y, edge_index))

    print('finished creating edges and clusters')
    return cluster_data
