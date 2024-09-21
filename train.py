import torch
from tqdm import tqdm
from dataloader import get_dataloader
from network import CNN
from torch import nn
from torchinfo import summary
import argparse

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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2
    )
    parser.add_argument(
        "--channels",
        type=tuple,
        default=(32, 64, 64)
    )
    parser.add_argument(
        "--kernels",
        type=tuple,
        default=(7, 5, 3)
    )
    parser.add_argument(
        "--strides",
        type=tuple,
        default=(1, 1, 1)
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=128
    )

    args = parser.parse_args()

    print(args)

    model = CNN(
        channels=args.channels,
        kernels=args.kernels,
        strides=args.strides,
        img_size=args.img_size
    ).to(device)
    print(summary(model))
    dataloader = get_dataloader()
    optim = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    num_epochs = 10
        
    train(model, dataloader, optim, loss_fn, num_epochs)