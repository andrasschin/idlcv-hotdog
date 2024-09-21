import torch
from tqdm import tqdm
from dataloader import get_dataloader
from network import CNN
from torch import nn
from torchinfo import summary

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, dataloader, optim, loss_fn, num_epochs):
    for epoch in range(num_epochs):

        running_loss = 0.0
        running_accuracy = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for (image, target) in progress_bar:
            image, target = image.to(device), target.to(device)
            optim.zero_grad()
            # target = target.float() # TODO: rewrite in dataloader
            pred = model(image)
            loss = loss_fn(pred, target)

            running_accuracy += (pred.argmax(dim=1) == target).sum()

            # Backward pass and optimize
            loss.backward()
            optim.step()

            # Track the loss
            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{running_loss / len(dataloader):.4f}'})

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Accuracy: {running_accuracy / (len(dataloader) * 64):.4f}')


if __name__ == "__main__":
    from rich import print

    model = CNN().to(device)
    print(summary(model))
    dataloader = get_dataloader()
    optim = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    num_epochs = 10
        
    train(model, dataloader, optim, loss_fn, num_epochs)