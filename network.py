import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, channels, kernels, strides, paddings, dropout_p, img_size):
        super().__init__()
        assert len(channels) == len(kernels) and len(kernels) == len(
            strides
        ), "Channels, kernels and strides must have same length."

        # Initial image size to calculate the output dimensions after each layer
        final_resolution = img_size
        
        self.cnn = nn.ModuleList()
        
        # First convolutional layer
        self.cnn.append(nn.Conv2d(in_channels=3,
                                   out_channels=channels[0],
                                   kernel_size=kernels[0],
                                   stride=strides[0],
                                   padding=paddings[0]))
        self.cnn.append(nn.ReLU())

        # Loop through the rest of the channels
        for i in range(len(channels) - 1):
            # Add batch normalization, conv, ReLU, and dropout for each layer
            self.cnn.append(nn.BatchNorm2d(channels[i]))
            if i!=0:
                self.cnn.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Max pooling after batch norm
            self.cnn.append(nn.Conv2d(in_channels=channels[i],
                                       out_channels=channels[i + 1],
                                       kernel_size=kernels[i + 1],
                                       stride=strides[i + 1],
                                       padding=paddings[i + 1]))
            self.cnn.append(nn.ReLU())
            self.cnn.append(nn.Dropout2d(p=dropout_p))
            
            # Calculate the final resolution after this layer
            final_resolution = (final_resolution - kernels[i + 1] + 2 * paddings[i + 1]) // strides[i + 1]
            final_resolution //= 2  # Max pooling halves the resolution
            
        # The output of the last pooling layer needs to be calculated properly
        self.linear = nn.Linear(
            in_features=channels[-1] * final_resolution**2, out_features=2
        )

    def forward(self, x):
        for layer in self.cnn:
            x = layer(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.linear(x)

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