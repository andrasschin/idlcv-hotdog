import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, channels, kernels, strides, img_size):
        super().__init__()
        assert len(channels) == len(kernels) and len(kernels) == len(strides), "Channels, kernels and strides must have same length."

        final_resolution = img_size
        for i, (k, s) in enumerate(zip(kernels, strides)):
            final_resolution = (final_resolution-k+1)/s
            assert final_resolution.is_integer(), "Resolution is not a whole number."
            final_resolution = int(final_resolution)
            print(f"[{i}] Layer resolution: {final_resolution}x{final_resolution}")
        final_resolution = int(final_resolution)

        self.cnn = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=channels[0], kernel_size=kernels[0], stride=strides[0]),
            nn.ReLU()
        ])
        
        for i in range(len(channels)-1):    
            self.cnn.extend([
                #nn.BatchNorm2d(channels[i]),
                nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=kernels[i+1], stride=strides[i+1]),
                nn.ReLU()
            ])

        self.linear = nn.Linear(in_features=channels[-1]*final_resolution**2, out_features=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.cnn:
            x = layer(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.linear(x)
        # Remove softmax because we have CrossEntropyLoss
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