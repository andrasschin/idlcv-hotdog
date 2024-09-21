import torch
from tqdm import tqdm
from dataloader import get_dataloader
from network import CNN
from torchinfo import summary
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using {device} device')
if not os.path.exists("outputs"):
    os.mkdir("outputs")
today = datetime.today().strftime('%m-%d-%H-%M')


def train(model, train_dataloader, test_dataloader, optim, loss_fn, num_epochs, batch_size=64):
    epochs = list(range(num_epochs))
    accuracies_train = []
    accuracies_test = []

    for epoch in range(num_epochs):
        model.train()
        running_loss_train = 0.0
        running_accuracy_train = 0.0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for (image, target) in progress_bar:
            image, target = image.to(device), target.to(device)
            optim.zero_grad()
 
            pred = model(image)
            loss = loss_fn(pred, target)

            running_accuracy_train += (pred.argmax(dim=1) == target).sum()

            loss.backward()
            optim.step()

            running_loss_train += loss.item()
            progress_bar.set_postfix({'Loss': f'{running_loss_train / len(train_dataloader):.4f}'})

        # print(f'[TRAIN] Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss_train / len(train_dataloader):.4f}')
        print(f'[TRAIN] Epoch [{epoch + 1}/{num_epochs}], Accuracy: {running_accuracy_train / (len(train_dataloader) * batch_size):.4f}')

        accuracies_train.append((running_accuracy_train / (len(train_dataloader) * batch_size)).cpu().numpy())

        # Evaluate
        model.eval()
        with torch.no_grad():
            running_loss_test = 0.0
            running_accuracy_test = 0.0
            for (image, target) in test_dataloader:
                image, target = image.to(device), target.to(device)
                pred = model(image)
                loss = loss_fn(pred, target)

                running_accuracy_test += (pred.argmax(dim=1) == target).sum()
                running_loss_test += loss.item()

            # print(f'[VAL] Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss_test / len(test_dataloader):.4f}')
            print(f'[VAL] Epoch [{epoch + 1}/{num_epochs}], Accuracy: {running_accuracy_test / (len(test_dataloader) * batch_size):.4f}')

        model.train()
        with open(f'outputs/results_{today}.txt', 'a') as f:
            f.write(f'Epoch: [{epoch+1}]\t')
            f.write(f'Train Loss: {running_loss_train / len(train_dataloader):.4f}\t')
            f.write(f'Train Accuracy: {running_accuracy_train / (len(train_dataloader) * batch_size):.4f}\t')
            f.write(f'Validation Loss: {running_loss_test / len(test_dataloader):.4f}\t')
            f.write(f'Validation Accuracy: {running_accuracy_test / (len(test_dataloader) * batch_size):.4f}\n')

        accuracies_test.append((running_accuracy_test / (len(test_dataloader) * batch_size)).cpu().numpy())

    #Plot accuracies
    plt.plot(epochs, accuracies_train, label='Train')
    plt.plot(epochs, accuracies_test, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    # Ticks for every second epoch
    plt.xticks(epochs[::2])
    plt.legend()
    plt.savefig(f'outputs/accuracies_{today}.png')


def eval(model, test_dataloader, loss_fn, batch_size=64):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        running_accuracy = 0.0
        for (image, target) in test_dataloader:
            image, target = image.to(device), target.to(device)
            pred = model(image)
            loss = loss_fn(pred, target)

            running_accuracy += (pred.argmax(dim=1) == target).sum()
            running_loss += loss.item()

        # print(f'[TEST] Loss: {running_loss / len(test_dataloader):.4f}')
        print(f'[TEST] Accuracy: {running_accuracy / (len(test_dataloader) * batch_size):.4f}')

    with open(f'outputs/results_{today}.txt', 'a') as f:
        f.write(f'----------------------\n')
        f.write(f'Loss: {running_loss / len(test_dataloader):.4f}\n')
        f.write(f'Accuracy: {running_accuracy / (len(test_dataloader) * batch_size):.4f}\n')

    print(f'Finished evaluation, results saved to results_{today}.txt')

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
        default=1e-3
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=[32, 64, 64, 64]
    )
    parser.add_argument(
        "--kernels",
        type=int,
        nargs="+",
        default=[7, 6, 3, 3]
    )
    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=[2, 2, 2, 1]
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

    train_dataloader = get_dataloader(batch_size=args.batch_size)
    test_dataloader = get_dataloader(train=False, batch_size=args.batch_size)

    optim = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    

    with open(f'outputs/results_{today}.txt', 'w') as f:
        f.write(f'{summary(model)}\n')
        f.write(f'Training parameters:\n')
        f.write(f'Batch Size: {args.batch_size}\n')
        f.write(f'Epochs: {args.epochs}\n')
        f.write(f'Learning Rate: {args.lr}\n')
        f.write(f'Channels: {args.channels}\n')
        f.write(f'Kernels: {args.kernels}\n')
        f.write(f'Strides: {args.strides}\n')
        f.write(f'Image Size: {args.img_size}\n')
        f.write(f'----------------------\n')

        resolution = args.img_size
        for i, (k, s) in enumerate(zip(args.kernels, args.strides)):
            resolution = (resolution-k+1)/s
            assert resolution.is_integer(), "Resolution is not a whole number."
            resolution = int(resolution)
            f.write(f"[{i}] Layer resolution: {resolution}x{resolution}\n")

        f.write(f'----------------------\n')

        
    train(model, train_dataloader, test_dataloader, optim, loss_fn, args.epochs, batch_size=args.batch_size)
    eval(model, test_dataloader, loss_fn, batch_size=args.batch_size)