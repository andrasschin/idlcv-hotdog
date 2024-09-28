import torch
from tqdm import tqdm
from dataloader import get_dataset
from network import CNN
from torchinfo import summary
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import StepLR

from PIL import Image
import numpy as np  

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
if not os.path.exists("outputs"):
    os.mkdir("outputs")
today = datetime.today().strftime('%m-%d-%H-%M')


def train(model, train_dataloader, val_dataloader, optim, loss_fn, num_epochs):
    scheduler = StepLR(optim, step_size=20, gamma=0.06)
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
                {"Loss": f"{running_loss_train / len(train_dataloader):.3f}"}
            )

        train_accuracy = n_correct_classifications / total_samples
        train_loss = running_loss_train / total_samples
        accuracies_train.append(train_accuracy)
        # print(f"[TRAIN] Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.3f}")
        print(
            f"[TRAIN] Epoch [{epoch + 1}/{num_epochs}], Accuracy: {train_accuracy:.3f}"
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

        # print(f"[VAL] Epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss:.3f}")
        print(f"[VAL] Epoch [{epoch + 1}/{num_epochs}], Accuracy: {val_accuracy:.3f}")

        with open(f'outputs/results_{today}.txt', 'a') as f:
            f.write(f'Epoch: [{epoch+1}]\t')
            f.write(f'Train Loss: {running_loss_train / len(train_dataloader):.4f}\t')
            f.write(f'Train Accuracy: {train_accuracy}\t')
            f.write(f'Validation Loss: {running_loss_val / len(val_dataloader):.4f}\t')
            f.write(f'Validation Accuracy: {val_accuracy}\n')

    # Plot accuracies
    epochs = list(range(1, num_epochs + 1))
    plt.plot(epochs, accuracies_train, label='Train')
    plt.plot(epochs, accuracies_val, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1) 
    plt.xticks(epochs[::2])
    plt.legend()
    plt.savefig(f'outputs/accuracies_{today}.png')
    scheduler.step()



def test(model, test_dataloader, loss_fn, device, save_dir):
    model.eval()
    all_labels = []
    all_preds = []

        # Create the directory to save images if it does not exist
    os.makedirs(save_dir, exist_ok=True)

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

            # Save the images with the predicted labels
            for i in range(image.size(0)):  # Loop through the batch
                img = image[i].cpu()  # Move image to CPU
                label = pred.argmax(dim=1)[i].item()  # Get the predicted label
                
                # Convert the image tensor to numpy array for saving
                img = img.permute(1, 2, 0).numpy()  # Change shape from (C, H, W) to (H, W, C)
                img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]

                # Convert to uint8 format for saving
                img = (img * 255).astype(np.uint8)

                # Define label strings
                predicted_label_str = 'hot_dog' if label == 0 else 'not_hot_dog'
                actual_label_str = 'hot_dog' if target[i].item() == 0 else 'not_hot_dog'

                # Create filename based on the predicted label
                filename = f"{save_dir}/img_{total_samples + i}_{predicted_label_str}.png"

                # Save image using PIL
                pil_image = Image.fromarray(img)
                pil_image.save(filename)

        test_accuracy = n_correct_classifications / total_samples
        test_loss = running_loss_test / total_samples

        print(f"[TEST] Accuracy: {test_accuracy:.3f}")
        
    with open(f'outputs/results_{today}.txt', 'a') as f:
        f.write(f'----------------------\n')
        f.write(f'Loss: {running_loss_test / len(test_dataloader):.4f}\n')
        f.write(f'Accuracy: {test_accuracy }\n')

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
        default=30,
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, nargs="+", default=[32, 64, 64, 64])
    parser.add_argument("--kernels", type=int, nargs="+", default=[7, 6, 3, 3])
    parser.add_argument("--strides", type=int, nargs="+", default=[2, 2, 2, 1])
    parser.add_argument("--paddings", type=int, nargs="+", default=[0, 0, 0, 0])
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)

    args = parser.parse_args()
    print(args)

    ############################ Model Creation ############################
    model = CNN(
        channels=args.channels,
        kernels=args.kernels,
        strides=args.strides,
        paddings=args.paddings,
        img_size=args.img_size,
        dropout_p=args.dropout
    ).to(device)
       
    ############################ Dataset Creation ############################
    train_dataset, val_dataset  = get_dataset("train",  image_size=args.img_size, do_aug=True)
    test_dataset = get_dataset("test", image_size=args.img_size, do_aug=False)
    
    ############################ Data Loaders ############################
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=3)
    val_dataloader  = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=3)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=3)
    
    ############################ Optimizer-Loss Function ############################
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    #optim =torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    ############################ Logging ############################
    with open(f'outputs/results_{today}.txt', 'w',encoding='utf-8') as f:
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

    try:
        train(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                optim=optim,
                loss_fn=loss_fn,
                num_epochs=args.epochs,
            )
        test(model=model, test_dataloader=test_dataloader, loss_fn=loss_fn,device=device,save_dir='outputs/saved_images')
    except KeyboardInterrupt:
        test(model=model, test_dataloader=test_dataloader, loss_fn=loss_fn,device=device,save_dir='outputs/saved_images')