import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, channels, kernels, strides, img_size):
        super().__init__()
        
        assert len(channels) == len(kernels) and len(kernels) == len(strides), \
            "Channels, kernels, and strides must have the same length."

        final_resolution = img_size
        for i, (k, s) in enumerate(zip(kernels, strides)):
            final_resolution = (final_resolution - k + 1) / s
            assert final_resolution.is_integer(), f"Resolution after layer {i} is not a whole number."
            final_resolution = int(final_resolution)
            print(f"[{i}] Layer resolution: {final_resolution}x{final_resolution}")
        final_resolution = int(final_resolution)

        layers = []
        layers.append(nn.Conv2d(in_channels=3, out_channels=channels[0], kernel_size=kernels[0], stride=strides[0]))
        layers.append(nn.ReLU())
        
        for i in range(1, len(channels)):
            layers.append(nn.Conv2d(in_channels=channels[i-1], out_channels=channels[i], kernel_size=kernels[i], stride=strides[i]))
            layers.append(nn.ReLU())

        self.cnn = nn.Sequential(*layers)
        self.linear = nn.Linear(in_features=channels[-1] * final_resolution * final_resolution, out_features=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  
        x = self.linear(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    'Create a CNN model with the following parameters:'
    'channels = Define multiple layers'
    'kernels = Kernel sizes for each layer'
    'strides = Strides for each layer'
    'img_size = Input image size'
    'Batch of 64 images, each 3x128x128'
    
    x = torch.randn(64, 3, 128, 128)
    model = CNN(
        channels=(32, 64),
        kernels=(3, 3),
        strides=(1, 1),
        img_size=128
    )
    y = model(x)
    print(f"Output shape: {y.shape}")
