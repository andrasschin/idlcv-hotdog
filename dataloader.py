import glob
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

class HotdogDataset(Dataset):
    def __init__(self, subset, transform, image_paths=None, name_to_label=None):
        assert subset in ["train", "validation", "test"]
        self.transform = transform
        self.image_paths = image_paths
        self.name_to_label = name_to_label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]

        X = self.transform(image)  # Apply transformation

        return X, y


def get_dataset(subset, image_size, do_aug, data_path="dataset\\hotdog_nothotdog\\"):

    # Set up transformations
    if do_aug:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomChoice([
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
            ]),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        # Define a transform for the training set (no augmentation)
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    
    # Define a transform for the validation set (no augmentation)
    base_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    data_path = os.path.join(data_path, subset)

    # Load all image paths and labels
    image_classes = [
        os.path.split(d)[1] for d in glob.glob(data_path + "/*") if os.path.isdir(d)
    ]
    image_classes.sort()
    name_to_label = {c: id for id, c in enumerate(image_classes)}
    all_image_paths = glob.glob(data_path + "/*/*.jpg")

    # Split the dataset if the subset is 'train'
    if subset == "train":
        train_paths, val_paths = train_test_split(all_image_paths, test_size=0.3, random_state=42)
        train_dataset = HotdogDataset(subset='train', transform=train_transform, image_paths=train_paths, name_to_label=name_to_label)
        val_dataset = HotdogDataset(subset='validation', transform=base_transform, image_paths=val_paths, name_to_label=name_to_label)
        return train_dataset, val_dataset
    else:
        return HotdogDataset(subset=subset, transform=base_transform, image_paths=all_image_paths, name_to_label=name_to_label)
 

if __name__ == "__main__":
    from rich import print

    batch_size = 64
    image_size = 128
    
    train_dataset, val_dataset  = get_dataset("train", image_size=image_size, do_aug=False)
    test_dataset = get_dataset("test", image_size=128, do_aug=False)

    next_train = next(iter(train_dataset))
    next_val = next(iter(val_dataset))
    next_test = next(iter(test_dataset))

    print("Length of train set: ", len(train_dataset) * batch_size)
    print("Length of validation set: ", len(val_dataset) * batch_size)
    print("Length of test set: ", len(test_dataset) * batch_size)
    
    print("Train Data shape: ", next_train[0].shape)
    print("Validation Data shape: ", next_val[0].shape)
    print("Test Data shape: ", next_test[0].shape)