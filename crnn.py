import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as transforms


class CRNN(nn.Module):
    def __init__(self, n_class=128, rnn_hidden_size=512, n_rnn_layers=3):
        super(CRNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Added fourth convolutional layer
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(512)



        # GRU layer will be initialized dynamically based on input size
        self.gru = None  
        self.fc = None  
        self.initialized = False  

        self.rnn_hidden_size = rnn_hidden_size
        self.n_rnn_layers = n_rnn_layers
        self.n_class = n_class

    def forward(self, x):
        # First convolutional block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 1))

        # Second convolutional block
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 1))

        # Third convolutional block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 1))

        # Fourth convolutional block (newly added)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 1))


        # Prepare input for GRU
        b, c, h, w = x.size()  # (Batch, Channels, Height, Width)
        x = x.permute(0, 3, 1, 2).contiguous()  # (Batch, Width, Channels, Height)
        x = x.view(b, w, -1)  # Flatten channels and height: (Batch, Width, Features)

        # Initialize GRU and FC layers dynamically based on input size
        if not self.initialized:
            # print("input_size:", input_size)
            input_size = x.size(-1)
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=self.rnn_hidden_size,
                num_layers=self.n_rnn_layers,
                batch_first=True,
                bidirectional=True
            ).to(x.device)
            self.fc = nn.Linear(2 * self.rnn_hidden_size, self.n_class).to(x.device)
            self.initialized = True

        # Pass through GRU
        x, _ = self.gru(x)

        # Pass through the fully connected layer
        x = self.fc(x)
        x = torch.sigmoid(x)  # Apply sigmoid for probabilities

        return x