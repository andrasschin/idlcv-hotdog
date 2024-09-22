import glob
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image


class HotdogDataset(Dataset):
    def __init__(
        self, train, transform, data_path="/work3/s233084/datasets/hotdog_nothotdog"
    ):
        self.transform = transform
        data_path = os.path.join(data_path, "train" if train else "test")
        image_classes = [
            os.path.split(d)[1] for d in glob.glob(data_path + "/*") if os.path.isdir(d)
        ]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + "/*/*.jpg")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        return X, y

def get_dataloader(train=True, image_size=128, batch_size=64, resize=True, rotate=True, normalize=True):
    # only ad transforms if they are true
    transform_list = []
    if resize:
        transform_list.append(transforms.Resize((image_size, image_size)))
    if rotate:
        transform_list.append(transforms.RandomRotation(5))
    if normalize:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    # Convert a color image to grayscale and normalize the color range to [0,1].
    transform_list.append(transforms.ToTensor())

    transform = transforms.Compose(transform_list)
    dataset = HotdogDataset(train=train, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=3
    )

    return dataloader



if __name__ == "__main__":
    from rich import print

    batch_size = 64
    image_size = 128
    train_loader = get_dataloader(train=True, batch_size=batch_size, image_size=image_size)
    test_loader = get_dataloader(train=False, batch_size=batch_size, image_size=image_size)

    next_train = next(iter(train_loader))
    next_test = next(iter(test_loader))

    print("Length of train set: ", len(train_loader) * batch_size)
    print("Length of test set: ", len(test_loader) * batch_size)
    print("Data shape: ", next_train[0].shape)
    print("Label shape: ", next_train[1].shape)
