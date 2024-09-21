import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2), # 128x128 -> 61x61
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=6, stride=2), # 61x61 -> 33x33
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2), # 33x33 -> 15x15
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1), # 15x15 -> 13x13
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1), # 13x13 -> 11x11
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1), # 11x11 -> 9x9
            nn.ReLU()
        )

        self.linear = nn.Linear(in_features=128*7*7, out_features=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.network(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.linear(x)
        x = self.softmax(x)

        return x

if __name__ == "__main__":
    x = torch.randn(64, 3, 128, 128)
    model = CNN()
    y = model(x)
    print(y.shape)