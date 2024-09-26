import torch
import torch.nn as nn
from torchvision import models

class FeatureExtractor(nn.Module):
    def __init__(self, num_classes=2):
        super(FeatureExtractor, self).__init__()
        
        # Load a pre-trained ResNet model
        self.base_model = models.resnet18(pretrained=True)
        
        # Remove the final classification layer (the fully connected layer)
        self.base_model = nn.Sequential(*(list(self.base_model.children())[:-1]))  # Keeps everything except the last layer
        
        # Flattening layer to convert the output into a 1D vector
        self.flatten = nn.Flatten()
        
        # New classifier layer
        self.classifier = nn.Linear(self.base_model[-1].in_features, num_classes)

    def forward(self, x):
        # Pass the input through the pre-trained model
        x = self.base_model(x)
        x = self.flatten(x)  # Flatten the output
        x = self.classifier(x)  # Classify using the new classifier
        return x

if __name__ == "__main__":
    model = FeatureExtractor(num_classes=2)
    x = torch.randn(64, 3, 224, 224)  # Example input size for ResNet
    y = model(x)
    print(y.shape)  # Output shape should be (64, num_classes)