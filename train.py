import torch
from tqdm import tqdm
from dataloader import get_dataset
from network import CNN
import argparse
from datetime import datetime
import os
from torch.utils.data import DataLoader
from utils import (
    plot_accuracies,
    write_results,
    write_config,
    write_test_results,
    plot_confusion_matrix,
)
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import StepLR

import warnings

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
print(f"Using {device} device")
if not os.path.exists("outputs"):
    os.mkdir("outputs")
os.mkdir("outputs/" + now)


def train(model, train_dataloader, val_dataloader, optim, loss_fn, num_epochs):
    scheduler = StepLR(optim, step_size=30, gamma=0.05)
    accuracies_train = []
    accuracies_val = []
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss_train = 0.0
        n_correct_classifications = 0.0
        total_samples = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for image, target in progress_bar:
            image, target = image.to(device), target.to(device)
            optim.zero_grad()

            pred = model(image)
            loss = loss_fn(pred, target)

            n_correct_classifications += (pred.argmax(dim=1) == target).sum().item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optim.step()

            running_loss_train += loss.item()
            total_samples += target.shape[0]
            progress_bar.set_postfix(
                {"Loss": f"{running_loss_train / len(train_dataloader):.4f}"}
            )

        train_accuracy = n_correct_classifications / total_samples
        train_loss = running_loss_train / total_samples
        accuracies_train.append(train_accuracy)
        # print(f"[TRAIN] Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}")
        print(
            f"[TRAIN] Epoch [{epoch + 1}/{num_epochs}], Accuracy: {train_accuracy:.4f}"
        )

        # Evaluate
        model.eval()
        with torch.no_grad():
            running_loss_val = 0.0
            n_correct_classifications = 0.0
            total_samples = 0.0
            for image, target in val_dataloader:
                image, target = image.to(device), target.to(device)
                pred = model(image)
                loss = loss_fn(pred, target)

                n_correct_classifications += (pred.argmax(dim=1) == target).sum().item()
                running_loss_val += loss.item()

                total_samples += target.shape[0]

        val_accuracy = n_correct_classifications / total_samples
        val_loss = running_loss_val / total_samples
        accuracies_val.append(val_accuracy)

        if best_val_accuracy < val_accuracy:
            best_val_accuracy = val_accuracy
            # torch.save(model, f"outputs/{now}/best_model_{val_accuracy:.1f}.pkl")

        # print(f"[VAL] Epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss:.4f}")
        print(f"[VAL] Epoch [{epoch + 1}/{num_epochs}], Accuracy: {val_accuracy:.4f}")

        write_results(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            now=now,
        )

        plot_accuracies(
            n_epochs=epoch,
            accuracies_train=accuracies_train,
            accuracies_val=accuracies_val,
            now=now,
        )

        scheduler.step()


def test(model, test_dataloader, loss_fn):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        running_loss_test = 0.0
        n_correct_classifications = 0.0
        total_samples = 0.0

        for image, target in test_dataloader:
            image, target = image.to(device), target.to(device)
            pred = model(image)
            loss = loss_fn(pred, target)

            n_correct_classifications += (pred.argmax(dim=1) == target).sum()
            running_loss_test += loss.item()

            all_preds.extend(pred.argmax(dim=1).cpu().numpy())
            all_labels.extend(target.cpu().numpy())

            total_samples += target.shape[0]

        test_accuracy = n_correct_classifications / total_samples
        test_loss = running_loss_test / total_samples

        # print(f'[TEST] Loss: {test_loss:.4f}')
        print(f"[TEST] Accuracy: {test_accuracy:.4f}")
    cm = confusion_matrix(all_labels, all_preds)
    write_test_results(test_loss=test_loss, test_accuracy=test_accuracy, now=now)
    plot_confusion_matrix(cm=cm, now=now)

    print(f"Finished evaluation, results saved to outputs/{now}.")


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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, nargs="+", default=[32, 64, 64, 64])
    parser.add_argument("--kernels", type=int, nargs="+", default=[7, 6, 3, 3])
    parser.add_argument("--strides", type=int, nargs="+", default=[2, 2, 2, 1])
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)

    args = parser.parse_args()

    print(args)

    model = CNN(
        channels=args.channels,
        kernels=args.kernels,
        strides=args.strides,
        img_size=args.img_size,
        dropout_p=args.dropout,
    ).to(device)

    dataset = get_dataset(train=True, image_size=args.img_size)

    # split test val dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    val_dataset.dataset.do_aug = False

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=3
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=3
    )

    test_dataloader = DataLoader(
        get_dataset(train=False, image_size=args.img_size, do_aug=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=3,
    )

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    write_config(cfg=args, model=model, now=now)

    train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optim=optim,
        loss_fn=loss_fn,
        num_epochs=args.epochs,
    )
    test(model=model, test_dataloader=test_dataloader, loss_fn=loss_fn)
