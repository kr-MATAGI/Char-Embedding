import torch
import torch.nn as nn
import torch.nn.functional as F

#====================================================
class CharCNN(nn.Module):
#====================================================
    def __init__(self):
        super(CharCNN, self).__init__()

        self.num_features = 5
        self.drop_prob = 0.1

        self.conv1 = nn.Sequential(
            nn.Conv1d(self.num_features, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.drop_prob)
        )

        self.fc3 = nn.Linear(1024, 4)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        # collapse
        x = x.view(x.size(0), -1)
        # linear layer
        x = self.fc1(x)
        # linear layer
        x = self.fc2(x)
        # linear layer
        x = self.fc3(x)
        # output layer
        x = self.log_softmax(x)

        return x