import torch
from torch import nn


class CategoricalDecoder(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, dropout_prob=0.4):
        super(CategoricalDecoder, self).__init__()

        # Shared layers for all categorical features
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

        # Branch for categorical features:
        # Each categorical feature gets its own output layer with softmax activation
        self.cat_fc1 = nn.Linear(hidden_size2, 5)  # Categorical feature with 5 classes
        self.cat_fc2 = nn.Linear(hidden_size2, 3)  # Categorical feature with 7 classes
        self.cat_fc3 = nn.Linear(hidden_size2, 4)  # Categorical feature with 4 classes

        self.softmax = nn.Softmax(dim=-1)  # For multi-class categorical features

    def forward(self, x):
        # Shared representation layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)

        # Categorical branch outputs
        cat_output1 = self.softmax(self.cat_fc1(x))  # Categorical feature with 5 classes
        cat_output2 = self.softmax(self.cat_fc2(x))  # Categorical feature with 7 classes
        cat_output3 = self.softmax(self.cat_fc3(x))  # Categorical feature with 4 classes

        # Concatenate all categorical outputs into one tensor
        cat_output = torch.cat([
            cat_output1,
            cat_output2,
            cat_output3
        ], dim=-1)

        return cat_output
