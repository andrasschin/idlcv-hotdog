import torch
import torch.nn as nn
import torch.nn.functional as F
from res_block import ResidualBlock

class CNN(nn.Module):
    def __init__(self, channels, kernels, strides, dropout_p, img_size, n_residuals):
        super(CNN, self).__init__()
        assert len(channels) == len(kernels) and len(kernels) == len(strides), "Channels, kernels, and strides must have the same length."

        # Calculate final output size after convolutions
        final_resolution = img_size
        for i, (k, s) in enumerate(zip(kernels, strides)):
            final_resolution = (final_resolution - k + 1) // s
            assert final_resolution > 0, "Resolution must be greater than 0."
        self.final_resolution = final_resolution

        # Initial CNN layers
        self.cnn_1 = nn.Sequential(
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(in_channels=3, out_channels=channels[0], kernel_size=kernels[0], stride=strides[0]),
            nn.ReLU()
        )

        # Residual blocks
        self.residuals = nn.ModuleList([
            ResidualBlock(in_channels=channels[0], out_channels=channels[0], kernels=kernels[1], strides=strides[1], dropout_p=dropout_p, n_residuals=n_residuals)
        ])
        self.residuals.extend([
            ResidualBlock(in_channels=channels[0], out_channels=channels[0], kernels=kernels[1], strides=strides[1], dropout_p=dropout_p, n_residuals=n_residuals)
            for _ in range(n_residuals)
        ])

        # Further CNN layers
        self.cnn_2 = nn.Sequential(
            nn.Dropout2d(p=dropout_p),
            nn.BatchNorm2d(channels[0]),
            nn.Conv2d(in_channels=channels[0], out_channels=channels[-1], kernel_size=kernels[-1], stride=strides[-1]),
            nn.ReLU()
        )

        # Linear layer for output
        self.linear = nn.Linear(in_features=25088, out_features=2)

    def forward(self, x):
        # Initial layers
        x = self.cnn_1(x)

        # Residual blocks
        for i, res_block in enumerate(self.residuals):
            x = res_block(x)

        # Further layers
        x = self.cnn_2(x)

        # Flatten and pass through the linear layer
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.linear(x)

        return x
