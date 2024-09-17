import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as transforms

class CRNN(nn.Module):
    def __init__(self, n_mels=229, n_class=128, rnn_hidden_size=128, n_rnn_layers=2, dropout=0.3):
        super(CRNN, self).__init__()

        # Mel Spectrogram extraction with log scaling and clipping
        self.mel_extractor = transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=2048,
            hop_length=160,
            f_min=0.,
            f_max=8000,
            n_mels=n_mels,
            power=2.0,
            normalized=True
        )

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # GRU layer initialized later based on actual input size
        self.gru = None  # GRU will be initialized dynamically
        self.fc = None  # Fully connected layer initialized dynamically
        self.initialized = False  # Flag to check if layers are initialized

        self.rnn_hidden_size = rnn_hidden_size
        self.n_rnn_layers = n_rnn_layers
        self.n_class = n_class

    def forward(self, audio):
        # Extract mel spectrogram
        x = self.mel_extractor(audio)  # (B, Freq, Time)

        # Apply log scaling and clipping for numerical stability
        x = torch.log10(torch.clamp(x, min=1e-10))

        # Pass through the first convolutional layer
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 1))  # Pooling only in the frequency domain

        # Pass through the second convolutional layer
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 1))

        # Pass through the third convolutional layer
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 1))

        # Prepare input for GRU
        b, c, h, w = x.size()  # (Batch, Channels, Height, Width)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, W, C, H) - Time becomes the second dimension
        x = x.view(b, w, -1)  # Flatten channels and height for GRU (B, W, C * H)

        # Dynamically initialize the GRU and FC layers after we know the input size
        if not self.initialized:
            input_size = x.size(-1)
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=self.rnn_hidden_size,
                num_layers=self.n_rnn_layers,
                batch_first=True,
                bidirectional=True
            ).to(x.device)  # Ensure the GRU is on the same device as the input
            self.fc = nn.Linear(2 * self.rnn_hidden_size, self.n_class).to(x.device)  # Fully connected layer
            self.initialized = True

        # Pass through GRU
        x, _ = self.gru(x)

        # Pass through the fully connected layer
        x = self.fc(x)
        x = torch.sigmoid(x)  # Apply sigmoid for probabilities

        return x
