import pickle
import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from gat_autoencoder.data_preprocessing import add_gaussian_noise
from gat_autoencoder.decoder_model import MyDecoder
from gat_autoencoder.edge_index import create_clusters, create_edge_index_for_full_data

from gat_autoencoder.encoder_model import MyGATv2Encoder
from gat_autoencoder.ssl_models import NeighbourResidual, NeighbourDiversity
from gat_autoencoder.binary_evaluation import evaluate_binary_imputation, print_binary_evaluation_results
from utils.plot_utils import plot_curve
from utils.utils import build_optimizer


def compute_neighbour_median_labels(A_eff, node_features, num_features):
    """
    Compute median of numerical features for each node's effective neighbors (memory-efficient).

    Args:
        A_eff: [num_nodes, num_nodes] - effective attention matrix (sparse, with only top neighbors)
        node_features: [num_nodes, total_features] - all node features
        num_features: int - number of numerical features

    Returns:
        median_labels: [num_nodes, num_features] - median of numerical features across neighbors
    """
    device = A_eff.device
    num_nodes = A_eff.shape[0]
    numerical_features = node_features[:, :num_features]  # [num_nodes, num_features]

    median_labels = torch.zeros(num_nodes, num_features, device=device)

    # Process in batches to avoid huge memory allocation
    batch_size = 1000
    for batch_start in range(0, num_nodes, batch_size):
        batch_end = min(batch_start + batch_size, num_nodes)
        batch_size_actual = batch_end - batch_start

        # Get neighbor mask for this batch
        neighbor_mask_batch = A_eff[batch_start:batch_end] > 0  # [batch_size, num_nodes]

        # Expand features for this batch only
        features_expanded = numerical_features.unsqueeze(0).expand(batch_size_actual, -1, -1)
        mask_expanded = neighbor_mask_batch.unsqueeze(2)

        # Mask out non-neighbors
        masked_features = torch.where(mask_expanded, features_expanded, torch.tensor(float('nan'), device=device))

        # Compute median
        median_labels[batch_start:batch_end] = torch.nanmedian(masked_features, dim=1)[0]

        # Fallback for nodes with no neighbors
        no_neighbors = ~neighbor_mask_batch.any(dim=1)
        if no_neighbors.any():
            median_labels[batch_start:batch_end][no_neighbors] = numerical_features[batch_start:batch_end][no_neighbors]

    return median_labels


def compute_residual_labels(node_features, median_labels, num_features):
    """
    Compute residual labels: difference between node features and neighborhood median.
    residual_if = x_if - median(x_jf for j in neighbors of i)

    Args:
        node_features: [num_nodes, total_features] - current node features (with imputations)
        median_labels: [num_nodes, num_features] - median of neighbors for each feature
        num_features: int - number of numerical features

    Returns:
        residual_labels: [num_nodes, num_features] - residuals for each node and feature
    """
    numerical_features = node_features[:, :num_features]  # [num_nodes, num_features]
    residual_labels = numerical_features - median_labels  # [num_nodes, num_features]
    return residual_labels


def compute_iqr_labels(A_eff, node_features, num_features):
    """
    Compute IQR (Interquartile Range) of numerical features for each node's effective neighbors (memory-efficient).
    IQR = Q3 - Q1 (75th percentile - 25th percentile)

    Args:
        A_eff: [num_nodes, num_nodes] - effective attention matrix (sparse, with only top neighbors)
        node_features: [num_nodes, total_features] - all node features
        num_features: int - number of numerical features

    Returns:
        iqr_labels: [num_nodes, num_features] - IQR of numerical features across neighbors
    """
    device = A_eff.device
    num_nodes = A_eff.shape[0]
    numerical_features = node_features[:, :num_features]  # [num_nodes, num_features]

    iqr_labels = torch.zeros(num_nodes, num_features, device=device)

    # Process in batches to avoid huge memory allocation
    batch_size = 1000
    for batch_start in range(0, num_nodes, batch_size):
        batch_end = min(batch_start + batch_size, num_nodes)
        batch_size_actual = batch_end - batch_start

        # Get neighbor mask for this batch
        neighbor_mask_batch = A_eff[batch_start:batch_end] > 0  # [batch_size, num_nodes]

        # Expand features for this batch only
        features_expanded = numerical_features.unsqueeze(0).expand(batch_size_actual, -1, -1)
        mask_expanded = neighbor_mask_batch.unsqueeze(2)

        # Mask out non-neighbors
        masked_features = torch.where(mask_expanded, features_expanded, torch.tensor(float('nan'), device=device))

        # Compute Q1 and Q3
        q1_labels = torch.nanquantile(masked_features, 0.25, dim=1)
        q3_labels = torch.nanquantile(masked_features, 0.75, dim=1)

        # Compute IQR
        iqr_labels[batch_start:batch_end] = q3_labels - q1_labels

        # Fallback for nodes with no neighbors
        no_neighbors = ~neighbor_mask_batch.any(dim=1)
        if no_neighbors.any():
            iqr_labels[batch_start:batch_end][no_neighbors] = 0

    return iqr_labels


def compute_effective_attention(edge_index, attention_weights, num_nodes, gamma1=1.0, gamma2=0.5, gamma3=0.25):
    """
    Compute effective attention matrix with multi-hop connections (sparse implementation).
    A_eff = gamma1 * A + gamma2 * A^2 + gamma3 * A^3

    Args:
        edge_index: [2, num_edges] - edge indices
        attention_weights: [num_edges] - attention weights
        num_nodes: int - total number of nodes
        gamma1, gamma2, gamma3: weights for 1-hop, 2-hop, 3-hop

    Returns:
        A_eff: [num_nodes, num_nodes] - effective attention matrix (sparse)
    """
    device = edge_index.device

    # Build sparse attention matrix A
    indices = edge_index  # [2, num_edges]
    values = attention_weights  # [num_edges]
    A_sparse = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=device)

    # Sparse matrix multiplication for A^2 and A^3
    A2_sparse = torch.sparse.mm(A_sparse, A_sparse)  # A^2 (2-hop)
    A3_sparse = torch.sparse.mm(A2_sparse, A_sparse)  # A^3 (3-hop)

    # Combine sparse matrices with weights
    # Note: Need to coalesce to merge duplicate indices
    A_eff = (gamma1 * A_sparse + gamma2 * A2_sparse + gamma3 * A3_sparse).coalesce()

    # Convert to dense only at the end (for median filtering)
    A_eff_dense = A_eff.to_dense()

    return A_eff_dense


def get_model(dataset, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_features = args.num_features
    cat_feature_sizes = getattr(args, 'cat_feature_sizes', None)

    # Calculate total input dimensions
    total_cat_features = sum(cat_feature_sizes) if cat_feature_sizes else 0
    in_channels = num_features + total_cat_features

    latent_dim = getattr(args, 'latent_dim', 10)
    neighbour_attention = getattr(args, 'neighbour_attention', False)
    encoder = MyGATv2Encoder(
        in_channels=in_channels,
        latent_dim=latent_dim,
        return_attention_weights=neighbour_attention
    ).to(device)

    decoder = MyDecoder(
        input_size=latent_dim,
        hidden_size1=28,
        hidden_size2=12,
        num_features=num_features,
        cat_feature_sizes=cat_feature_sizes,
        dropout_prob=0.2
    ).to(device)

    # SSL models
    ssl_residual = None
    ssl_diversity = None

    if getattr(args, 'use_neighbour_residual', False):
        ssl_residual = NeighbourResidual(
            embedding_dim=latent_dim,
            num_features=num_features,
            hidden_dim=32
        ).to(device)

    if getattr(args, 'use_neighbour_diversity', False):
        ssl_diversity = NeighbourDiversity(
            embedding_dim=latent_dim,
            num_features=num_features,
            hidden_dim=32
        ).to(device)

    print(encoder)
    print(decoder)
    if ssl_residual:
        print('SSL Residual:', ssl_residual)
    if ssl_diversity:
        print('SSL Diversity:', ssl_diversity)

    return encoder, decoder, ssl_residual, ssl_diversity


def eval_model_categorical(data, path, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('from file')
    encoder = torch.load(path + 'encoder.pt')
    decoder = torch.load(path + 'decoder.pt')

    encoder.eval()
    decoder.eval()

    # Create edge index for the entire dataset
    edge_index_full = create_edge_index_for_full_data(data.df)

    cat_feature_sizes = getattr(args, 'cat_feature_sizes', None)
    has_categorical = cat_feature_sizes and len(cat_feature_sizes) > 0

    with torch.no_grad():
        edge_index = edge_index_full.to(device)

        # Forward pass through encoder and decoder
        if getattr(args, 'neighbour_attention', False):
            z, _ = encoder(data.x.to(device), edge_index)
        else:
            z = encoder(data.x.to(device), edge_index)
        num_output, cat_outputs = decoder(z)

        # Evaluate numerical features
        target_num = data.y[:, :args.num_features].to(device)
        pred_num = num_output.cpu().numpy()
        actual_num = target_num.cpu().numpy()

        # Use test_mask if available, otherwise evaluate all
        if hasattr(data, 'test_mask'):
            mask = data.test_mask
            pred_num_masked = pred_num[mask]
            actual_num_masked = actual_num[mask]
        else:
            pred_num_masked = pred_num
            actual_num_masked = actual_num

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        mse = mean_squared_error(actual_num_masked, pred_num_masked)
        mae = mean_absolute_error(actual_num_masked, pred_num_masked)
        r2 = r2_score(actual_num_masked, pred_num_masked)

        print(f"\nNumerical Features Evaluation:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.6f}")

        # Evaluate categorical features (only if they exist)
        if has_categorical:
            target_tensor = data.y.to(device)
            offset = args.num_features

            target_cats = []
            pred_cats = []
            for i, n_classes in enumerate(cat_feature_sizes):
                target_cat = target_tensor[:, offset:offset+n_classes].argmax(dim=-1)
                pred_cat = cat_outputs[i].argmax(dim=-1)
                target_cats.append(target_cat)
                pred_cats.append(pred_cat)
                offset += n_classes

            for i, n_classes in enumerate(cat_feature_sizes):
                correct = (pred_cats[i] == target_cats[i]).sum().item()
                total = target_cats[i].size(0)
                accuracy = correct / total if total > 0 else 0

                print(f"\nAccuracy for Categorical Group {i+1} ({n_classes} Classes): {accuracy:.4f}")

                cm = confusion_matrix(target_cats[i].cpu().numpy(), pred_cats[i].cpu().numpy())
                print(f"Confusion Matrix for Categorical Feature {i+1}:\n{cm}")
        else:
            print("\nNo categorical features to evaluate.")

    return


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


def train_model_categorical(data, path, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder, decoder, ssl_residual, ssl_diversity = get_model(data, args)

    # Optimizer for main model and SSL models
    params = list(encoder.parameters()) + list(decoder.parameters())
    if ssl_residual is not None:
        params += list(ssl_residual.parameters())
    if ssl_diversity is not None:
        params += list(ssl_diversity.parameters())
    scheduler, opt = build_optimizer(args, params)

    # Train
    train_loss_list = []
    train_loss_list2 = []
    test_loss_cat_list = []

    # Loss functions
    criterion_numerical = torch.nn.MSELoss()
    cat_feature_sizes = getattr(args, 'cat_feature_sizes', None)
    has_categorical = cat_feature_sizes and len(cat_feature_sizes) > 0
    criterion_categorical = [torch.nn.CrossEntropyLoss() for _ in cat_feature_sizes] if has_categorical else []

    # Create clusters: single cluster for benchmark datasets, multi-cluster for custom data
    use_single_cluster = getattr(args, 'binary_eval', False)  # Benchmark datasets use single cluster

    if use_single_cluster:
        # Single cluster mode (for Cora, Citeseer, etc.)
        print("Using single-cluster mode (benchmark dataset)")
        edge_index_test = data.edge_index.to(device)
        cluster_data = [(data.x, data.edge_index)]
        num_clusters = 1
    else:
        # Multi-cluster mode (for custom datasets with spatial grouping)
        print("Using multi-cluster mode (custom dataset)")
        cluster_data = create_clusters(data.df)
        edge_index_test = create_edge_index_for_full_data(data.df)
        num_clusters = len(cluster_data)

    # SSL: Store median and IQR labels per cluster (updated every 5 epochs)
    cluster_median_labels = [None] * num_clusters
    cluster_iqr_labels = [None] * num_clusters

    epoch = 0
    while epoch < args.epochs:
        epoch += 1
        start_time = time.time()

        encoder.train()
        decoder.train()
        if ssl_residual is not None:
            ssl_residual.train()
        if ssl_diversity is not None:
            ssl_diversity.train()

        train_loss_sum = 0

        # Train on each cluster
        for cluster_idx, (cluster_df, edge_index) in enumerate(cluster_data):
            opt.zero_grad()  # Zero out gradients before starting cluster iterations

            cluster_x = cluster_df.to(device)
            edge_index = edge_index.to(device)

            # Add Gaussian noise to the numerical input part only
            numerical_part = cluster_x[:, :10]
            categorical_part = cluster_x[:, 10:]
            noisy_numerical_part = add_gaussian_noise(numerical_part)
            cluster_x_noisy = torch.cat([noisy_numerical_part, categorical_part], dim=1)

            # Forward pass through encoder
            median_labels = None
            if getattr(args, 'neighbour_attention', False):
                z, (edge_index_att, attention_weights) = encoder(cluster_x_noisy, edge_index)

                # Compute A_eff at epoch 0 and every 5 epochs
                if epoch == 0 or (epoch % 5 == 0):
                    attention_avg = attention_weights.mean(dim=1)
                    cluster_size = cluster_x.shape[0]

                    A_eff_cluster = compute_effective_attention(
                        edge_index_att,
                        attention_avg,
                        cluster_size,
                        gamma1=getattr(args, 'gamma1', 1.0),
                        gamma2=getattr(args, 'gamma2', 0.5),
                        gamma3=getattr(args, 'gamma3', 0.25)
                    )

                    # Keep only neighbors above median per row, set rest to 0
                    thresholds = A_eff_cluster.median(dim=1, keepdim=True)[0]
                    A_eff_cluster = torch.where(A_eff_cluster >= thresholds, A_eff_cluster, torch.zeros_like(A_eff_cluster))

                    # Compute and store SSL labels (updated when A_eff changes)
                    if ssl_residual is not None:
                        median_labels = compute_neighbour_median_labels(A_eff_cluster, cluster_x, args.num_features)
                        cluster_median_labels[cluster_idx] = median_labels

                    if ssl_diversity is not None:
                        iqr_labels = compute_iqr_labels(A_eff_cluster, cluster_x, args.num_features)
                        cluster_iqr_labels[cluster_idx] = iqr_labels
            else:
                z = encoder(cluster_x_noisy, edge_index)

            # Forward pass through decoder
            num_output, cat_outputs = decoder(z)

            # Numerical loss
            target_num = cluster_x[:, :args.num_features]
            loss_num = criterion_numerical(num_output, target_num)

            # Categorical losses (only if categorical features exist)
            loss_cat = 0
            if has_categorical:
                offset = args.num_features
                for i, n_classes in enumerate(cat_feature_sizes):
                    target_cat = cluster_x[:, offset:offset+n_classes].argmax(dim=-1)
                    loss_cat += criterion_categorical[i](cat_outputs[i], target_cat)
                    offset += n_classes

            # SSL Residual Loss
            loss_ssl_residual = 0
            if ssl_residual is not None and cluster_median_labels[cluster_idx] is not None:
                # Compute residual labels: x_i - median(neighbors)
                residual_labels = compute_residual_labels(
                    cluster_x,
                    cluster_median_labels[cluster_idx],
                    args.num_features
                )

                # Get non-masked numerical features as input
                numerical_features = cluster_x[:, :args.num_features]  # [num_nodes, num_features]

                # Predict residuals from raw features
                residual_pred = ssl_residual(numerical_features)  # [num_nodes, num_features]

                # Only compute loss on non-masked features
                # Assuming we have access to mask for this cluster (if available)
                # For now, compute loss on all features
                criterion_ssl = torch.nn.MSELoss()
                loss_ssl_residual = criterion_ssl(residual_pred, residual_labels)

            # SSL Diversity Loss
            loss_ssl_diversity = 0
            if ssl_diversity is not None and cluster_iqr_labels[cluster_idx] is not None:
                # Get IQR labels for this cluster
                iqr_labels = cluster_iqr_labels[cluster_idx]

                # Get non-masked numerical features as input
                numerical_features = cluster_x[:, :args.num_features]  # [num_nodes, num_features]

                # Predict IQR from raw features
                iqr_pred = ssl_diversity(numerical_features)  # [num_nodes, num_features]

                # Compute loss on all features (or apply mask if available)
                criterion_ssl = torch.nn.MSELoss()
                loss_ssl_diversity = criterion_ssl(iqr_pred, iqr_labels)

            # Total loss
            ssl_weight_residual = getattr(args, 'ssl_weight_residual', 0.1)
            ssl_weight_diversity = getattr(args, 'ssl_weight_diversity', 0.1)
            loss = loss_num + loss_cat + ssl_weight_residual * loss_ssl_residual + ssl_weight_diversity * loss_ssl_diversity

            loss.backward()  # Accumulate gradients across clusters
            train_loss_sum += loss.item()
            train_loss_list2.append(loss.item())
            opt.step()  # Update model parameters after accumulating gradients for all clusters
            # print(f"Training loss epoch: {loss.item()}")

        avg_train_loss = train_loss_sum / num_clusters
        train_loss_list.append(avg_train_loss)
        print(f'Epoch: {epoch}, Train Loss: {avg_train_loss:.6f}')

        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training time: {training_time:.2f} seconds")
        print('---------------------------')

        # Validation step (every 3 epochs) on the entire graph
        if (epoch % 3) == 0:
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                if getattr(args, 'neighbour_attention', False):
                    z, _ = encoder(data.x.to(device), edge_index_test.to(device))
                else:
                    z = encoder(data.x.to(device), edge_index_test.to(device))
                num_output, cat_outputs = decoder(z)

                # Numerical loss
                target_num = data.y[:, :args.num_features].to(device)
                loss_num = criterion_numerical(num_output, target_num)

                # Categorical losses (only if categorical features exist)
                loss_cat = 0
                if has_categorical:
                    offset = args.num_features
                    for i, n_classes in enumerate(cat_feature_sizes):
                        target_cat = data.y[:, offset:offset+n_classes].argmax(dim=-1).to(device)
                        loss_cat += criterion_categorical[i](cat_outputs[i], target_cat)
                        offset += n_classes

                test_loss = loss_num + loss_cat

                if has_categorical:
                    print(f'Test Loss: {test_loss:.6f} (Num: {loss_num:.6f}, Cat: {loss_cat:.6f})')
                else:
                    print(f'Test Loss: {test_loss:.6f} (Num: {loss_num:.6f})')
                test_loss_cat_list.append(test_loss.item())

            # Update learning rate using ReduceLROnPlateau scheduler
            scheduler.step(test_loss.item())

        if epoch == -3:
            break

    obj = dict()
    obj['args'] = args
    obj['loss'] = dict()
    obj['loss']['train_loss'] = train_loss_list
    obj['loss']['train_loss_batch'] = train_loss_list2
    obj['loss']['test_loss_cat'] = test_loss_cat_list

    pickle.dump(obj, open(path + 'result.pkl', "wb"))

    torch.save(encoder, path + 'encoder.pt')
    torch.save(decoder, path + 'decoder.pt')

    plot_curve(obj['loss'], path + 'loss.png', keys=None,
               clip=True, label_min=True, label_end=True)

    # Binary evaluation for benchmark datasets (if enabled)
    if getattr(args, 'binary_eval', False):
        print("\n" + "="*60)
        print("Running Binary Evaluation on Test Set")
        print("="*60)

        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            # Forward pass on full dataset
            if getattr(args, 'neighbour_attention', False):
                z, _ = encoder(data.x.to(device), edge_index_test.to(device))
            else:
                z = encoder(data.x.to(device), edge_index_test.to(device))

            num_output, _ = decoder(z)

            # Get predictions and ground truth
            y_pred = num_output.cpu().numpy()
            y_true = data.y.cpu().numpy()
            test_mask = data.test_mask

            # Evaluate with Recall@K and NDCG@K
            k_values = getattr(args, 'eval_k_values', [5, 10, 20])
            results = evaluate_binary_imputation(y_true, y_pred, test_mask, k_values=k_values)

            # Print results
            print_binary_evaluation_results(results)

            # Save results
            obj['binary_eval'] = results
            pickle.dump(obj, open(path + 'result.pkl', "wb"))


def run(data, path, args):
    if args.is_eval:
        eval_model_categorical(data, path, args)

    else:
        train_model_categorical(data, path, args)
