import torch
import torch.nn as nn


class Decoder4(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_prob=0.4):
        super(Decoder4, self).__init__()

        # Adding more layers for a deeper decoder
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size2)
        self.fc4 = nn.Linear(hidden_size2, hidden_size1)
        self.fc5 = nn.Linear(hidden_size1, output_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # To ensure the output is in [0, 1] range

        # Dropout layer to avoid over-fitting
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc5(x))
        return x


class NumericalDecoder(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_prob=0.4):
        super(NumericalDecoder, self).__init__()

        # Fully connected layers for numerical feature reconstruction
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size2)
        self.fc4 = nn.Linear(hidden_size2, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.sigmoid = nn.Sigmoid()  # Ensures the output is between 0 and 1

    def forward(self, x):
        # Pass through fully connected layers with ReLU and dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        # Final output with Sigmoid activation to ensure values between 0 and 1
        x = self.sigmoid(self.fc4(x))
        return x

