import pickle
import time

import joblib
import numpy as np
import torch
import torch_geometric
from matplotlib import pyplot as plt
from torch import nn

from gat_autoencoder.v2.decoder_model import Decoder1, EnhancedDecoder
from gat_autoencoder.v2.encoder_model import MyGATv2EncoderVAE
from gat_autoencoder.v2.load_data import split_into_clusters, create_cluster_edge_index, add_gaussian_noise
from utils.plot_utils import plot_curve
from utils.utils import build_optimizer


def get_model(dataset, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = MyGATv2EncoderVAE().to(device)

    decoder2 = EnhancedDecoder(input_size=8, hidden_size1=32, hidden_size2=16, output_size=9,
                               dropout_prob=0.2)

    print(encoder)
    print(decoder2)
    return encoder, decoder2


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

    imputed_data = data.x.clone().to(device)
    imask = torch.tensor(data.test_mask, dtype=torch.bool).to(device)
    original_2 = imputed_data[:, 9:]
    original_1 = imputed_data[:, :9]

    num_iterations = 10
    for _ in range(num_iterations):
        mu, log_var = encoder(imputed_data, data.edge_index.to(device))
        z = reparameterize(mu, log_var)
        imputed_eval = decoder(z)

        imputed_data = torch.where(imask, imputed_eval, original_1)
        imputed_data = torch.concat([imputed_data, original_2], dim=1)

    imputed_eval = imputed_data[:, :9]
    data.y = data.y[:, :9]

    pred_val = imputed_eval.detach().cpu().numpy()
    actual_val = data.y.detach().cpu().numpy()

    abs_diff = np.abs(pred_val - actual_val)
    error_perc = column_wise_average(abs_diff, actual_val, imask.cpu().numpy())

    print(error_perc)

    return


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


def train_model(data, path, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder, decoder = get_model(data, args)

    scheduler, opt = build_optimizer(args, list(encoder.parameters()) + list(decoder.parameters()))

    # Split the data into clusters
    num_clusters = 20  # Number of clusters
    clusters = split_into_clusters(data.x, num_clusters)
    cluster_s = data.x.shape[0] // num_clusters

    # Train
    Train_loss = []
    Test_loss = []

    epoch = 0
    while epoch < args.epochs:
        epoch += 1
        start_time = time.time()

        encoder.train()
        decoder.train()

        train_loss_sum = 0
        opt.zero_grad()  # Zero out gradients before starting cluster iterations

        # Train on each cluster
        for i, cluster_x in enumerate(clusters):
            cluster_size = cluster_x.shape[0]
            edge_index = create_cluster_edge_index(cluster_size)

            # Add Gaussian noise to the input cluster
            cluster_x_noisy = add_gaussian_noise(cluster_x)

            # Move data to device
            x = cluster_x_noisy.to(device)
            edge_index = edge_index.to(device)

            start_idx = i * cluster_s
            end_idx = min((i + 1) * cluster_s, data.y.shape[0])
            y = data.y[start_idx:end_idx, :].to(device)

            # Forward pass through encoder and reparameterize trick
            mu, log_var = encoder(x, edge_index)
            z = reparameterize(mu, log_var)

            # Forward pass through decoder
            imputed = decoder(z)

            # Compute reconstruction loss
            criterion = torch.nn.MSELoss()
            recon_loss = criterion(imputed, y[:, :9])

            # Compute KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / cluster_size

            # Total VAE loss
            beta = min(1.0, epoch / 100)  # Linearly increase beta until it reaches 1
            loss = recon_loss + beta * kl_loss

            # loss = recon_loss + kl_loss

            loss.backward()  # Accumulate gradients across clusters
            train_loss_sum += loss.item()

        opt.step()  # Update model parameters after accumulating gradients for all clusters
        print('Loss relation:', kl_loss.item() / recon_loss.item())

        avg_train_loss = train_loss_sum / num_clusters
        Train_loss.append(avg_train_loss)
        print(f'Epoch: {epoch}')
        print(f'Train Loss: {avg_train_loss}')

        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training time: {training_time:.2f}")

        # Validation step (every 3 epochs) on the entire graph
        if (epoch % 3) == 0:
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                # Use the entire graph for evaluation
                mu, log_var = encoder(data.x.to(device), data.edge_index.to(device))
                z = reparameterize(mu, log_var)
                imputed = decoder(z)

                # Compute test loss for the entire graph
                pred_test = imputed[data.test_mask]
                label_test = data.y[:, :9][data.test_mask].to(device)

                test_loss = criterion(pred_test, label_test).item()
                print(f'Test Loss: {test_loss}')
                Test_loss.append(test_loss)

            # Update learning rate using ReduceLROnPlateau scheduler
            scheduler.step(test_loss)

        if epoch == -3:
            break

    obj = dict()
    obj['args'] = args
    obj['loss'] = dict()
    obj['loss']['train_loss'] = Train_loss
    obj['loss']['test_loss'] = Test_loss

    pickle.dump(obj, open(path + 'result.pkl', "wb"))

    torch.save(encoder, path + 'encoder.pt')
    torch.save(decoder, path + 'decoder.pt')

    plot_curve(obj['loss'], path + 'loss.png', keys=None,
               clip=True, label_min=True, label_end=True)


def run(data, path, args):
    if args.is_eval:
        eval_model(data, path, args)

    else:
        train_model(data, path, args)
