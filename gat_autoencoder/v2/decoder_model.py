import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Decoder, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=30),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=30, out_features=45),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=45, out_features=45),
            nn.ReLU()
        )
        self.layer4 = nn.Linear(in_features=45, out_features=output_size)  # output layer with linear activation

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class Decoder1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function for output layer

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


class EnhancedDecoder(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_prob=0.4):
        super(EnhancedDecoder, self).__init__()

        # Adding more layers for a deeper decoder
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size2)
        self.fc4 = nn.Linear(hidden_size2, hidden_size1)
        self.fc5 = nn.Linear(hidden_size1, output_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # To ensure the output is in [0, 1] range

        # Dropout layer to avoid overfitting
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


class Decoder3(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(Decoder3, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Softmax activation function for output layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        x1 = self.sigmoid(x[:, :10])
        x2 = self.softmax(x[:, 10:15])
        x3 = self.softmax(x[:, 15:23])
        x4 = self.softmax(x[:, 23:32])
        x5 = self.softmax(x[:, 32:36])

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return x
