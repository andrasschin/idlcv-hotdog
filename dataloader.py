import glob
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class HotdogDataset(Dataset):
    def __init__(
        self,
        subset,
        transform,
        image_size,
        data_path="/work3/s233084/datasets/hotdog_nothotdog",
    ):
        assert subset in ["train", "validation", "test"]
        self.transform = transform
        data_path = os.path.join(data_path, subset)
        image_classes = [
            os.path.split(d)[1] for d in glob.glob(data_path + "/*") if os.path.isdir(d)
        ]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + "/*/*.jpg")
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]

        X = self.transform(image)

        return X, y


def get_dataset(
    subset,
    image_size,
    do_aug,
):
    augmentation_transforms = transforms.RandomChoice(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomErasing(
                p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False
            ),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomAffine(0, shear=5),
            transforms.RandomAffine(0, scale=(0.8, 1.2)),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
        ]
    )

    if do_aug:
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                augmentation_transforms,
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    dataset = HotdogDataset(subset=subset, transform=transform, image_size=image_size)

    return dataset


if __name__ == "__main__":
    from rich import print

    batch_size = 64
    image_size = 128
    train_loader = get_dataset(train=True, batch_size=batch_size, image_size=image_size)
    test_loader = get_dataset(train=False, batch_size=batch_size, image_size=image_size)

    next_train = next(iter(train_loader))
    next_test = next(iter(test_loader))

    print("Length of train set: ", len(train_loader) * batch_size)
    print("Length of test set: ", len(test_loader) * batch_size)
    print("Data shape: ", next_train[0].shape)
    print("Label shape: ", next_train[1].shape)
