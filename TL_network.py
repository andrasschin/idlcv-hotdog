import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
class ResNet50Model(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNet50Model, self).__init__()
        
        self.base_model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        in_features = self.base_model.fc.in_features
        
        # Remove the original fully connected layer
        self.base_model.fc = nn.Identity()  
        
        # Add custom layers
        self.fc1 = nn.Linear(in_features, 1000) 
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1000, num_classes)  

    def forward(self, x):
        x = self.base_model(x)  
        x = self.fc1(x) 
        x = self.dropout(x) 
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    x = torch.randn(64, 3, 128, 128) 
    model = ResNet50Model(num_classes=2) 
    y = model(x) 
    print(y.shape) 