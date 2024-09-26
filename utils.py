import matplotlib.pyplot as plt
from torchinfo import summary
import seaborn as sns

sns.set(style="whitegrid")


def write_config(cfg, model, now):
    with open(f"outputs/{now}/results.txt", "w") as f:
        f.write(f"{summary(model)}\n")
        f.write(f"Training parameters:\n")
        f.write(f"Batch Size: {cfg.batch_size}\n")
        f.write(f"Epochs: {cfg.epochs}\n")
        f.write(f"Learning Rate: {cfg.lr}\n")
        f.write(f"Channels: {cfg.channels}\n")
        f.write(f"Kernels: {cfg.kernels}\n")
        f.write(f"Strides: {cfg.strides}\n")
        f.write(f"Image Size: {cfg.img_size}\n")
        f.write(f"Image Size: {cfg.dropout}\n")
        f.write(f"----------------------\n")

        resolution = cfg.img_size
        for i, (k, s) in enumerate(zip(cfg.kernels, cfg.strides)):
            resolution = (resolution - k + 1) / s
            assert resolution.is_integer(), "Resolution is not a whole number."
            resolution = int(resolution)
            f.write(f"[{i}] Layer resolution: {resolution}x{resolution}\n")

        f.write(f"----------------------\n")


def write_results(epoch, train_loss, train_accuracy, val_loss, val_accuracy, now):
    with open(f"outputs/{now}/results.txt", "a") as f:
        f.write(f"Epoch: [{epoch+1}]\t")
        # f.write(f"Train Loss: {train_loss:.3f}\t")
        f.write(f"Train Accuracy: {train_accuracy:.3f}\t")
        # f.write(f"Validation Loss: {val_loss:.3f}\t")
        f.write(f"Validation Accuracy: {val_accuracy:.3f}\n")


def write_test_results(test_loss, test_accuracy, now):
    with open(f"outputs/{now}/results.txt", "a") as f:
        f.write(f"----------------------\n")
        f.write(f"Loss: {test_loss:.3f}\n")
        f.write(f"Accuracy: {test_accuracy:.3f}\n")


def plot_accuracies(n_epochs, accuracies_train, accuracies_val, now):
    epochs_list = list(range(n_epochs + 1))
    plt.figure()
    palette = sns.color_palette("husl", 2)
    plt.plot(
        epochs_list,
        accuracies_train,
        label="Train",
        marker="o",
        color=palette[0],
        linewidth=2,
    )
    plt.plot(
        epochs_list,
        accuracies_val,
        label="Validation",
        marker="o",
        color=palette[1],
        linewidth=2,
    )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks(epochs_list[::2])
    plt.legend()
    plt.savefig(f"outputs/{now}/accuracies.png")
    plt.close()


def plot_confusion_matrix(cm, test_accuracy, now):
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=["Not hotdog", "Hotdog"],
        yticklabels=["Not hotdog", "Hotdog"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (Test accuracy: {test_accuracy:.3f})")
    plt.savefig(f"outputs/{now}/confusion_matrix.png")
    plt.close()
