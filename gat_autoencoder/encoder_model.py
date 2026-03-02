import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.nn import BatchNorm1d


class MyGATv2Encoder(torch.nn.Module):
    def __init__(self, in_channels, latent_dim=10, return_attention_weights=False):
        super(MyGATv2Encoder, self).__init__()

        self.return_attention_weights = return_attention_weights
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # Dynamic layer sizes based on input
        hidden1 = max(int(in_channels * 0.8), latent_dim + 4)
        hidden2 = max(int(in_channels * 0.6), latent_dim + 2)
        hidden3 = max(latent_dim + 2, latent_dim + 1)

        # First GATv2 layer
        self.conv1 = GATv2Conv(in_channels=in_channels, out_channels=hidden1, heads=3, concat=True, dropout=0.3)
        self.bn1 = BatchNorm1d(hidden1 * 3)

        # Second GATv2 layer
        self.conv2 = GATv2Conv(in_channels=hidden1 * 3, out_channels=hidden2, heads=3, concat=True, dropout=0.3)
        self.bn2 = BatchNorm1d(hidden2 * 3)

        # Third GATv2 layer
        self.conv3 = GATv2Conv(in_channels=hidden2 * 3, out_channels=hidden3, heads=3, concat=True, dropout=0.3)
        self.bn3 = BatchNorm1d(hidden3 * 3)

        # Fourth GATv2 layer (output layer)
        self.conv4 = GATv2Conv(in_channels=hidden3 * 3, out_channels=latent_dim, heads=5, concat=False, dropout=0.3)
        self.bn4 = BatchNorm1d(latent_dim)

    def forward(self, x, edge_index):
        # First GATv2 layer with BatchNorm
        x1 = F.dropout(x, p=0.3, training=self.training)
        x1 = self.conv1(x1, edge_index)
        x1 = self.bn1(x1)
        x1 = F.elu(x1)

        # Second GATv2 layer with BatchNorm
        x2 = F.dropout(x1, p=0.3, training=self.training)
        x2 = self.conv2(x2, edge_index)
        x2 = self.bn2(x2)
        x2 = F.elu(x2)

        # Third GATv2 layer with BatchNorm
        x3 = F.dropout(x2, p=0.3, training=self.training)
        x3 = self.conv3(x3, edge_index)
        x3 = self.bn3(x3)
        x3 = F.elu(x3)

        # Fourth GATv2 layer with BatchNorm (last layer - use for attention)
        x4 = F.dropout(x3, p=0.3, training=self.training)
        if self.return_attention_weights:
            x4, (edge_index_att, attention_weights) = self.conv4(x4, edge_index, return_attention_weights=True)
        else:
            x4 = self.conv4(x4, edge_index)
        x4 = self.bn4(x4)

        if self.return_attention_weights:
            return x4, (edge_index_att, attention_weights)
        else:
            return x4
