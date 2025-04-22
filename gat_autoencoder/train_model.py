import pickle
import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, r2_score

from gat_autoencoder.categorical_model.edge_index import create_edge_index_for_full_data, create_clusters
from gat_autoencoder.decoder_model import NumericalDecoder
from gat_autoencoder.encoder_model import MyGATv2EncoderVAEOLD
from gat_autoencoder.v2.load_data import split_into_clusters, create_cluster_edge_index, add_gaussian_noise
from utils.plot_utils import plot_curve
from utils.utils import build_optimizer


def get_model(dataset, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = MyGATv2EncoderVAEOLD().to(device)

    # decoder2 = Decoder4(input_size=10, hidden_size1=32, hidden_size2=16, output_size=11,
    #                     dropout_prob=0.2)

    decoder = NumericalDecoder(input_size=10, hidden_size1=32, hidden_size2=16, output_size=10, dropout_prob=0.2)

    print(encoder)
    print(decoder)
    return encoder, decoder


def column_wise_average(abs_diff, actual_val, imask):
    result = []
    num_columns = abs_diff.shape[1]

    for col in range(num_columns):
        abs_diff_col = abs_diff[:, col]
        actual_val_col = actual_val[:, col]
        mask_col = imask[:, col]

        masked_abs_diff = abs_diff_col[mask_col]
        masked_actual_val = actual_val_col[mask_col]

        with np.errstate(divide='ignore', invalid='ignore'):
            division_result = np.divide(masked_abs_diff, masked_actual_val)

        valid_division_result = division_result[np.isfinite(division_result)]
        # valid_division_result = division_result[division_result != 0]

        if len(valid_division_result) > 0:
            average = np.mean(valid_division_result)
        else:
            average = 0

        result.append(average)

    return result


def eval_model(data, path, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('from file')
    encoder = torch.load(path + 'encoder.pt')
    decoder = torch.load(path + 'decoder.pt')

    encoder.eval()
    decoder.eval()

    edge_index_full = create_edge_index_for_full_data(data.df)

    imputed_data = data.x.clone().to(device)
    imask = torch.tensor(data.test_mask, dtype=torch.bool).to(device)
    original_1 = imputed_data[:, :10].detach().clone()  # Numerical features
    original_2 = imputed_data[:, 10:].detach().clone()  # Numerical features

    iter_mae_list = []
    num_iterations = 10
    for _ in range(num_iterations):
        mu, log_var = encoder(imputed_data, edge_index_full)
        z = reparameterize(mu, log_var)
        imputed_eval = decoder(z)

        pred = imputed_eval[imask].detach()
        real = original_1[imask].detach()
        mae = mean_absolute_error(real, pred)
        iter_mae_list.append(mae)


        # Update numerical values with the imputed ones for masked entries
        imputed_data = torch.where(imask, imputed_eval[:, :10], original_1)
        imputed_data = torch.concat([imputed_data, original_2], dim=1)

    # Numerical evaluation (first 11 columns)
    data.y1 = data.y[:, :10].detach().clone()

    pred_val = imputed_data.detach().cpu().numpy()
    actual_val = data.y1.detach().cpu().numpy()

    abs_diff = np.abs(pred_val - actual_val)
    error_perc = column_wise_average(abs_diff, actual_val, imask.cpu().numpy())

    print("Numerical Feature Error Percentage:", error_perc)

    t1 = pred_val[imask]
    t2 = actual_val[imask]
    mse = mean_squared_error(t2, t1)
    mae = mean_absolute_error(t2, t1)
    r2 = r2_score(t2, t1)
    print(mse, mae, r2)
    return


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


def train_model(data, path, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder, decoder = get_model(data, args)

    scheduler, opt = build_optimizer(args, list(encoder.parameters()) + list(decoder.parameters()))

    # Use custom clusters and edge creation
    cluster_data = create_clusters(data.df, data.df_y)
    edge_index_full = create_edge_index_for_full_data(data.df)

    # Train
    train_loss_list = []
    test_loss_num_list = []

    epoch = 0
    while epoch < args.epochs:
        epoch += 1
        start_time = time.time()

        encoder.train()
        decoder.train()

        train_loss_sum = 0

        # Train on each cluster
        for cluster_x, cluster_y, edge_index in cluster_data:
            opt.zero_grad()  # Zero out gradients before starting cluster iterations

            # Move cluster data to the device
            cluster_x = cluster_x.to(device)
            edge_index = edge_index.to(device)

            # Add Gaussian noise to the numerical input part only
            numerical_part = cluster_x[:, :10]

            noisy_numerical_part = add_gaussian_noise(numerical_part)

            cluster_x_noisy = torch.cat([noisy_numerical_part, cluster_x[:, 10:]], dim=1)

            # Forward pass through encoder and reparameterize trick
            mu, log_var = encoder(cluster_x_noisy, edge_index)
            z = reparameterize(mu, log_var)

            # Forward pass through decoder
            imputed = decoder(z)

            # Compute reconstruction loss for numerical features
            target_numerical = cluster_y[:, :10]  # Use numerical features as target
            criterion_numerical = torch.nn.MSELoss()
            recon_loss_numerical = criterion_numerical(imputed, target_numerical)

            # Compute KL divergence loss
            cluster_size = cluster_x.shape[0]
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / cluster_size

            # Total VAE loss
            beta = 0.1  # Adjust beta as needed
            loss = recon_loss_numerical + beta * kl_loss

            loss.backward()  # Accumulate gradients across clusters
            train_loss_sum += loss.item()

            opt.step()  # Update model parameters after accumulating gradients for all clusters

        avg_train_loss = train_loss_sum / len(cluster_data)
        train_loss_list.append(avg_train_loss)
        print(f'Epoch: {epoch}, Train Loss: {avg_train_loss:.6f}')

        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training time: {training_time:.2f} seconds")
        print('---------------------------')

        # if (epoch % 3) == 0:
        #     encoder.eval()
        #     decoder.eval()
        #     with torch.no_grad():
        #         # Forward pass on the entire dataset
        #         mu, log_var = encoder(data.x, edge_index_full)
        #         z = reparameterize(mu, log_var)
        #         imputed = decoder(z)
        #
        #         target_numerical = data.y[:, :10].to(device)  # Validation target (numerical only)
        #         recon_loss_numerical = criterion_numerical(imputed, target_numerical)
        #
        #         print(f'Validation Reconstruction Loss: {recon_loss_numerical:.6f}')
        #         scheduler.step(recon_loss_numerical.item())
        #         test_loss_num_list.append(recon_loss_numerical.item())

        if epoch == -3:
            break

    obj = dict()
    obj['args'] = args
    obj['loss'] = dict()
    obj['loss']['train_loss'] = train_loss_list
    # obj['loss']['test_loss_num'] = test_loss_num_list

    pickle.dump(obj, open(path + 'result.pkl', "wb"))

    torch.save(encoder, path + 'encoder.pt')
    torch.save(decoder, path + 'decoder.pt')

    plot_curve(obj['loss'], path + 'loss-m10.svg', keys=None,
               clip=True, label_min=True, label_end=True)


def run(data, path, args):
    if args.is_eval:
        eval_model(data, path, args)

    else:
        train_model(data, path, args)
