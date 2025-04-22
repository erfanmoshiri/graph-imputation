import pickle
import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from gat_autoencoder.categorical_model.categorical_decoder import CategoricalDecoder
from gat_autoencoder.categorical_model.categorical_encoder import MyGATv2EncoderCategorical
from gat_autoencoder.categorical_model.categorical_load_data import find_weight_for_cats
from gat_autoencoder.categorical_model.edge_index import create_clusters, create_edge_index_for_full_data
from gat_autoencoder.v2.load_data import split_into_clusters, create_cluster_edge_index, add_gaussian_noise
from utils.plot_utils import plot_curve
from utils.utils import build_optimizer


def get_model(dataset, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = MyGATv2EncoderCategorical().to(device)

    # decoder2 = Decoder4(input_size=10, hidden_size1=32, hidden_size2=16, output_size=11,
    #                     dropout_prob=0.2)

    decoder = CategoricalDecoder(input_size=10, hidden_size1=28, hidden_size2=12, dropout_prob=0.2)

    print(encoder)
    print(decoder)
    return encoder, decoder


def eval_model_categorical(data, path, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('from file')
    encoder = torch.load(path + 'encoder.pt')
    decoder = torch.load(path + 'decoder.pt')

    encoder.eval()
    decoder.eval()

    # Create edge index for the entire dataset
    edge_index_full = create_edge_index_for_full_data(data.df)

    with torch.no_grad():
        # Convert data to PyTorch tensor
        # data_x = torch.tensor(data.df, dtype=torch.float).to(device)
        edge_index = edge_index_full.to(device)

        # Forward pass through the encoder and decoder
        z = encoder(data.x.to(device), edge_index)
        imputed = decoder(z)

        # Split imputed output for evaluation
        imputed_categorical1 = imputed[:, :5]  # Categorical feature with 5 classes
        imputed_categorical2 = imputed[:, 5:8]  # Categorical feature with 7 classes
        imputed_categorical3 = imputed[:, 8:16]  # Categorical feature with 4 classes

        # Split target data similarly
        target_tensor = data.y.to(device)
        target_categorical1 = target_tensor[:, 10:15].argmax(dim=-1)
        target_categorical2 = target_tensor[:, 15:18].argmax(dim=-1)
        target_categorical3 = target_tensor[:, 18:22].argmax(dim=-1)

        # Convert predicted softmax outputs to class indices
        pred_categorical1 = imputed_categorical1.argmax(dim=-1)
        pred_categorical2 = imputed_categorical2.argmax(dim=-1)
        pred_categorical3 = imputed_categorical3.argmax(dim=-1)

        # Calculate the number of correct predictions and total samples for each category
        correct_cat1 = (pred_categorical1 == target_categorical1).sum().item()
        total_cat1 = target_categorical1.size(0)

        correct_cat2 = (pred_categorical2 == target_categorical2).sum().item()
        total_cat2 = target_categorical2.size(0)

        correct_cat3 = (pred_categorical3 == target_categorical3).sum().item()
        total_cat3 = target_categorical3.size(0)

        # Calculate accuracy for each categorical group
        cat1_accuracy = correct_cat1 / total_cat1 if total_cat1 > 0 else 0
        cat2_accuracy = correct_cat2 / total_cat2 if total_cat2 > 0 else 0
        cat3_accuracy = correct_cat3 / total_cat3 if total_cat3 > 0 else 0

        # Print group accuracies
        print("\nAccuracy for Categorical Group 1 (5 Classes):", cat1_accuracy)
        print("Accuracy for Categorical Group 2 (7 Classes):", cat2_accuracy)
        print("Accuracy for Categorical Group 3 (4 Classes):", cat3_accuracy)

        # Calculate confusion matrices for each categorical group
        cm1 = confusion_matrix(target_categorical1.cpu().numpy(), pred_categorical1.cpu().numpy())
        cm2 = confusion_matrix(target_categorical2.cpu().numpy(), pred_categorical2.cpu().numpy())
        cm3 = confusion_matrix(target_categorical3.cpu().numpy(), pred_categorical3.cpu().numpy())

        # Print confusion matrices
        print("\nConfusion Matrix for Categorical Feature 1 (5 Classes):\n", cm1)
        print("\nConfusion Matrix for Categorical Feature 2 (7 Classes):\n", cm2)
        print("\nConfusion Matrix for Categorical Feature 3 (4 Classes):\n", cm3)

    return


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


def train_model_categorical(data, path, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder, decoder = get_model(data, args)

    scheduler, opt = build_optimizer(args, list(encoder.parameters()) + list(decoder.parameters()))

    # Train
    train_loss_list = []
    train_loss_list2 = []
    test_loss_cat_list = []

    cat1_weights, cat2_weights, cat3_weights = find_weight_for_cats()
    cat1_weights_tensor = torch.tensor(cat1_weights.values, dtype=torch.float).to(device)
    cat2_weights_tensor = torch.tensor(cat2_weights.values, dtype=torch.float).to(device)
    cat3_weights_tensor = torch.tensor(cat3_weights.values, dtype=torch.float).to(device)
    # criterion_categorical1 = torch.nn.CrossEntropyLoss(weight=cat1_weights_tensor)
    # criterion_categorical2 = torch.nn.CrossEntropyLoss(weight=cat2_weights_tensor)
    # criterion_categorical3 = torch.nn.CrossEntropyLoss(weight=cat3_weights_tensor)
    criterion_categorical1 = torch.nn.CrossEntropyLoss()
    criterion_categorical2 = torch.nn.CrossEntropyLoss()
    criterion_categorical3 = torch.nn.CrossEntropyLoss()

    cluster_data = create_clusters(data.df)
    edge_index_test = create_edge_index_for_full_data(data.df)
    num_clusters = len(cluster_data)

    epoch = 0
    while epoch < args.epochs:
        epoch += 1
        start_time = time.time()

        encoder.train()
        decoder.train()

        train_loss_sum = 0

        # Train on each cluster
        for cluster_df, edge_index in cluster_data:
            opt.zero_grad()  # Zero out gradients before starting cluster iterations

            cluster_x = cluster_df.to(device)
            edge_index = edge_index.to(device)

            # Add Gaussian noise to the numerical input part only
            numerical_part = cluster_x[:, :10]
            categorical_part = cluster_x[:, 10:]
            noisy_numerical_part = add_gaussian_noise(numerical_part)
            cluster_x_noisy = torch.cat([noisy_numerical_part, categorical_part], dim=1)

            # Forward pass through encoder
            z = encoder(cluster_x_noisy, edge_index)

            # Forward pass through decoder
            imputed = decoder(z)

            # Split imputed output into categorical parts
            imputed_categorical1 = imputed[:, :5]  # Categorical feature with 5 classes
            imputed_categorical2 = imputed[:, 5:8]  # Categorical feature with 7 classes
            imputed_categorical3 = imputed[:, 8:12]  # Categorical feature with 4 classes

            # Split target data similarly
            target_categorical1 = cluster_x[:, 10:15].argmax(dim=-1)  # One-hot to index for 5 classes
            target_categorical2 = cluster_x[:, 15:18].argmax(dim=-1)  # One-hot to index for 7 classes
            target_categorical3 = cluster_x[:, 18:22].argmax(dim=-1)  # One-hot to index for 4 classes

            # Compute reconstruction loss for categorical features
            # criterion_categorical = torch.nn.CrossEntropyLoss()

            recon_loss_categorical1 = criterion_categorical1(imputed_categorical1, target_categorical1)
            recon_loss_categorical2 = criterion_categorical2(imputed_categorical2, target_categorical2)
            recon_loss_categorical3 = criterion_categorical3(imputed_categorical3, target_categorical3)

            # Combine categorical losses
            recon_loss_categorical = recon_loss_categorical1 + recon_loss_categorical2 + recon_loss_categorical3

            # Total loss (without KL divergence, as it's no longer a VAE)
            loss = recon_loss_categorical

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
                z = encoder(data.x.to(device), edge_index_test)
                imputed = decoder(z)

                # Split imputed output for evaluation
                imputed_categorical1 = imputed[:, :5]
                imputed_categorical2 = imputed[:, 5:8]
                imputed_categorical3 = imputed[:, 8:12]

                # Split target data similarly
                target_categorical1 = data.y[:, 10:15].argmax(dim=-1).to(device)
                target_categorical2 = data.y[:, 15:18].argmax(dim=-1).to(device)
                target_categorical3 = data.y[:, 18:22].argmax(dim=-1).to(device)

                # Compute test loss
                recon_loss_categorical1 = criterion_categorical1(imputed_categorical1, target_categorical1)
                recon_loss_categorical2 = criterion_categorical2(imputed_categorical2, target_categorical2)
                recon_loss_categorical3 = criterion_categorical3(imputed_categorical3, target_categorical3)

                recon_loss_categorical = recon_loss_categorical1 + recon_loss_categorical2 + recon_loss_categorical3

                print(f'test recon_loss_categorical: {recon_loss_categorical:.6f}')
                test_loss_cat_list.append(recon_loss_categorical.item())

            # Update learning rate using ReduceLROnPlateau scheduler
            scheduler.step(recon_loss_categorical.item())

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


def run(data, path, args):
    if args.is_eval:
        eval_model_categorical(data, path, args)

    else:
        train_model_categorical(data, path, args)
