import torch
from torch import nn


class MyDecoder(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_features, cat_feature_sizes=None, dropout_prob=0.4):
        super(MyDecoder, self).__init__()

        self.num_features = num_features
        self.cat_feature_sizes = cat_feature_sizes if cat_feature_sizes else []
        self.has_categorical = len(self.cat_feature_sizes) > 0

        # Shared base layers
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

        # Numerical branch
        self.num_fc1 = nn.Linear(hidden_size2, hidden_size2)
        self.num_fc2 = nn.Linear(hidden_size2, num_features)
        self.sigmoid = nn.Sigmoid()

        # Categorical branch (only if there are categorical features)
        if self.has_categorical:
            self.cat_fc_shared = nn.Linear(hidden_size2, hidden_size2)
            self.cat_heads = nn.ModuleList([nn.Linear(hidden_size2, n_classes) for n_classes in self.cat_feature_sizes])

    def forward(self, x):
        # Shared base representation
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        # Numerical branch
        x_num = self.relu(self.num_fc1(x))
        x_num = self.dropout(x_num)
        num_output = self.sigmoid(self.num_fc2(x_num))

        # Categorical branch (only if categorical features exist)
        if self.has_categorical:
            x_cat = self.relu(self.cat_fc_shared(x))
            x_cat = self.dropout(x_cat)
            cat_outputs = [head(x_cat) for head in self.cat_heads]
        else:
            cat_outputs = []

        return num_output, cat_outputs
