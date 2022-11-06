import torch
import torch.nn as nn
import torch.nn.functional as F

#====================================================
class CharCNN(nn.Module):
#====================================================
    def __init__(self,
                 vocab_dict,
                 vocab_size,
                 seq_len
                 ):
        super(CharCNN, self).__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = seq_len
        self.drop_prob = 0.1

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.max_seq_len * 3, out_channels=256,
                kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=256, out_channels=256,
                kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=256, out_channels=256,
                kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256,
                      kernel_size=3),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(11776, 1024),
            nn.ReLU(),
            nn.Dropout(self.drop_prob)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(self.drop_prob)
        )

    def forward(self, x):
        '''
            x : [batch_size, seq_len * 3, vocab_size]
        '''

        # Conv Net
        x = self.conv1(x) # [batch_size, output_ch, vocab_size]
        x = self.conv2(x) # [batch_size, output_ch, vocab_size]
        x = self.conv3(x)
        x = self.conv4(x)

        # Fully-Connected
        x_shape = x.shape
        x = self.fc1(x.view(x_shape[0], -1))
        x = self.fc2(x)

        return x