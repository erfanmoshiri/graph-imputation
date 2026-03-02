import torch
import torch.nn as nn


class NeighbourResidual(nn.Module):
    """
    SSL Model 2: Neighbour-Residual
    Shallow MLP for residual-based objective.
    """
    def __init__(self, embedding_dim, num_features, hidden_dim=32):
        super(NeighbourResidual, self).__init__()

        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_features)
        self.relu = nn.ReLU()

    def forward(self, z):
        """
        Args:
            z: [batch_size, embedding_dim] - node embeddings
        Returns:
            pred: [batch_size, num_features] - predicted residual values
        """
        x = self.relu(self.fc1(z))
        pred = self.fc2(x)
        return pred


class NeighbourDiversity(nn.Module):
    """
    SSL Model 3: Neighbour-Diversity
    Shallow MLP for diversity-based objective.
    """
    def __init__(self, embedding_dim, num_features, hidden_dim=32):
        super(NeighbourDiversity, self).__init__()

        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_features)
        self.relu = nn.ReLU()

    def forward(self, z):
        """
        Args:
            z: [batch_size, embedding_dim] - node embeddings
        Returns:
            pred: [batch_size, num_features] - predicted diversity values
        """
        x = self.relu(self.fc1(z))
        pred = self.fc2(x)
        return pred