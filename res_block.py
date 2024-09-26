import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernels, strides, dropout_p, n_residuals):
        super(ResidualBlock, self).__init__()
        self.n_residuals = n_residuals

        # Define layers for the residual block
        self.layers = nn.ModuleList()
        for _ in range(n_residuals):
            self.layers.append(nn.Sequential(
                nn.Dropout2d(p=dropout_p),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernels, stride=strides, padding=kernels // 2),
                nn.ReLU()
            ))
            in_channels = out_channels  # For subsequent layers

    def forward(self, x):
        res = x
        for layer in self.layers:
            x = layer(x)

        # Ensure the input and output shapes match
        if res.shape != x.shape:
            res = F.interpolate(res, size=x.shape[2:], mode='nearest')

        return F.relu(x + res)
