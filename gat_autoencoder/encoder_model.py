import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.nn import BatchNorm1d


# class MyGATv2(torch.nn.Module):
#     def __init__(self):
#         super(MyGATv2, self).__init__()
#
#         # First GATv2 layer
#         self.conv1 = GATv2Conv(in_channels=29, out_channels=16, heads=5, concat=True, dropout=0.2)
#
#         # Second GATv2 layer
#         self.conv2 = GATv2Conv(in_channels=16 * 5, out_channels=16, heads=5, concat=True, dropout=0.2)
#
#         # Final GATv2 layer
#         self.conv3 = GATv2Conv(in_channels=16 * 5, out_channels=8, heads=5, concat=False, dropout=0.2)
#
#     def forward(self, x, edge_index):
#         # x, edge_index = data.x, data.edge_index
#
#         # First layer
#         x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
#         x = self.conv1(x, edge_index)
#         x = torch.nn.functional.elu(x)
#
#         # Second layer
#         x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
#         x = self.conv2(x, edge_index)
#         x = torch.nn.functional.elu(x)
#
#         # Final layer
#         x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
#         x = self.conv3(x, edge_index)
#
#         return torch.sigmoid(x)


class MyGATv2EncoderVAEOLD(torch.nn.Module):
    def __init__(self):
        super(MyGATv2EncoderVAEOLD, self).__init__()

        # First GATv2 layer
        self.conv1 = GATv2Conv(in_channels=22, out_channels=18, heads=3, concat=True, dropout=0.3)
        self.bn1 = BatchNorm1d(18 * 3)

        # Second GATv2 layer with residual connection
        self.conv2 = GATv2Conv(in_channels=18 * 3, out_channels=14, heads=3, concat=True, dropout=0.3)
        self.bn2 = BatchNorm1d(14 * 3)

        # Third GATv2 layer with residual connection
        self.conv3 = GATv2Conv(in_channels=14 * 3, out_channels=12, heads=3, concat=True, dropout=0.3)
        self.bn3 = BatchNorm1d(12 * 3)

        # Fourth GATv2 layer
        self.conv4 = GATv2Conv(in_channels=12 * 3, out_channels=10, heads=5, concat=False, dropout=0.3)
        self.bn4 = BatchNorm1d(10)

        # Output layers for mean and log-variance of the latent space
        self.fc_mu = torch.nn.Linear(10, 10)  # Mean of the latent space
        self.fc_log_var = torch.nn.Linear(10, 10)  # Log variance of the latent space

    def forward(self, x, edge_index):
        # First GATv2 layer with BatchNorm
        x1 = F.dropout(x, p=0.3, training=self.training)
        x1 = self.conv1(x1, edge_index)
        x1 = self.bn1(x1)
        x1 = F.elu(x1)

        # Second GATv2 layer with residual connection and BatchNorm
        x2 = F.dropout(x1, p=0.3, training=self.training)
        x2 = self.conv2(x2, edge_index)
        x2 = self.bn2(x2)
        # x2 = F.elu(x2) + x1  # Residual connection with the first layer's output

        # Third GATv2 layer with residual connection and BatchNorm
        x3 = F.dropout(x2, p=0.3, training=self.training)
        x3 = self.conv3(x3, edge_index)
        x3 = self.bn3(x3)
        # x3 = F.elu(x3) + x2  # Residual connection with the second layer's output

        # Fourth GATv2 layer with BatchNorm
        x4 = F.dropout(x3, p=0.3, training=self.training)
        x4 = self.conv4(x4, edge_index)
        x4 = self.bn4(x4)

        # Compute mean and log-variance for latent space
        mu = self.fc_mu(x4)
        log_var = self.fc_log_var(x4)

        return mu, log_var
