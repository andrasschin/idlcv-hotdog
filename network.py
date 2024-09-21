import torch
import torch.nn as nn
from functools import reduce

class CNN(nn.Module):
    def __init__(self, channels, kernels, strides, img_size):
        super().__init__()
        assert len(channels) == len(kernels) and len(kernels) == len(strides), "Channels, kernels and strides must have same length."
        self.cnn = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=channels[0], kernel_size=kernels[0], stride=strides[0]),
            nn.ReLU()
        ])
        
        for i in range(len(channels)-1):
            self.cnn.extend([
                nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=kernels[i+1], stride=strides[i+1]),
                nn.ReLU()
            ])

        final_resolution = reduce(lambda res, ks: (res-ks[0]+1)/ks[1], zip(kernels, strides), img_size)
        assert final_resolution.is_integer(), "Resolution is not a whole number."
        final_resolution = int(final_resolution)
        print(f"Final resolution of the downsampled image is {final_resolution}x{final_resolution}")

        self.linear = nn.Linear(in_features=channels[-1]*final_resolution**2, out_features=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.cnn:
            x = layer(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.linear(x)
        x = self.softmax(x)

        return x

if __name__ == "__main__":
    x = torch.randn(64, 3, 128, 128)
    model = CNN(
        channels=(32),
        kernels=(3),
        strides=(1)
    )
    y = model(x)
    print(y.shape)