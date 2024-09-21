import torch
from tqdm import tqdm
from dataloader import get_dataloader
#from network import get_model
from torch import nn

device = "gpu" if torch.cuda.is_available() else "cpu"

#model = get_model().to(device)
#batch_size = 64
#dataloader = get_dataloader(batch_size=batch_size)
#optim = torch.optim.Adam(model.parameters())
#loss_fn = torch.nn.MSELoss()
#num_epochs = 10


def train(model, dataloader, optim, loss_fn, num_epochs):
    for epoch in range(num_epochs):

        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for (image, target) in progress_bar:
            #print(f"image shape: {image.shape}")
            #print(f"target shape: {target.shape}")
            #print(f"target dtype: {target.dtype}")
            optim.zero_grad()
            target = target.float() # TODO: rewrite in dataloader
            pred = model(image)
            #print(f"pred dtype: {pred.dtype}")
            loss = loss_fn(pred, target)

            # Backward pass and optimize
            loss.backward()
            optim.step()

            # Track the loss
            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{running_loss / len(dataloader):.4f}'})

        print(f'Epoch [{epoch + 1}/{epoch}], Loss: {running_loss / len(dataloader):.4f}')


if __name__ == "__main__":
    from rich import print


    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
            self.flatten = torch.nn.Flatten()
            self.fc = torch.nn.Linear(in_features=49152, out_features=1)

        def forward(self, x):
            x = self.conv1(x)
            #print(f"Shape after conv: {x.shape}")
            x = self.flatten(x)
            #print(f"Shape after flatten: {x.shape}")
            x = self.fc(x).squeeze(1)
            #print(f"Shape after fc: {x.shape}")
            return x
        
    model = SimpleModel()
    dataloader = get_dataloader()
    optim = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    num_epochs = 10
        
    train(model, dataloader, optim, loss_fn, num_epochs)