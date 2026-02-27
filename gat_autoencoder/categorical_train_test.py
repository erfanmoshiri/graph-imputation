import pickle
import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from gat_autoencoder.categorical_load_data import find_weight_for_cats, add_gaussian_noise
from gat_autoencoder.decoder_model import MyDecoder
from gat_autoencoder.edge_index import create_clusters, create_edge_index_for_full_data

from gat_autoencoder.encoder_model import MyGATv2Encoder
from utils.plot_utils import plot_curve
from utils.utils import build_optimizer


def get_model(dataset, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = MyGATv2Encoder().to(device)

    num_features = args.num_features
    cat_feature_sizes = getattr(args, 'cat_feature_sizes', None)

    decoder = MyDecoder(
        input_size=10,
        hidden_size1=28,
        hidden_size2=12,
        num_features=num_features,
        cat_feature_sizes=cat_feature_sizes,
        dropout_prob=0.2
    ).to(device)

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

    cat_feature_sizes = getattr(args, 'cat_feature_sizes', None)
    has_categorical = cat_feature_sizes and len(cat_feature_sizes) > 0

    with torch.no_grad():
        edge_index = edge_index_full.to(device)

        # Forward pass through encoder and decoder
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

    encoder, decoder = get_model(data, args)

    scheduler, opt = build_optimizer(args, list(encoder.parameters()) + list(decoder.parameters()))

    # Train
    train_loss_list = []
    train_loss_list2 = []
    test_loss_cat_list = []

    # Loss functions
    criterion_numerical = torch.nn.MSELoss()
    cat_feature_sizes = getattr(args, 'cat_feature_sizes', None)
    has_categorical = cat_feature_sizes and len(cat_feature_sizes) > 0
    criterion_categorical = [torch.nn.CrossEntropyLoss() for _ in cat_feature_sizes] if has_categorical else []

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

            # Total loss
            loss = loss_num + loss_cat

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


def run(data, path, args):
    if args.is_eval:
        eval_model_categorical(data, path, args)

    else:
        train_model_categorical(data, path, args)
