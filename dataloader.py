import glob
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class HotdogDataset(Dataset):
    def __init__(
        self, train, transform, data_path="dataset\\hotdog_nothotdog\\", do_aug=False, image_size=128, apply_all_transforms=False
    ):
        self.transform = transform
        data_path = os.path.join(data_path, "train" if train else "test")
        image_classes = [
            os.path.split(d)[1] for d in glob.glob(data_path + "/*") if os.path.isdir(d)
        ]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + "/*/*.jpg")
        self.do_aug = do_aug
        self.apply_all_transforms = apply_all_transforms
        self.basic_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((image_size, image_size))])
        self.image_size = image_size
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]

        if self.do_aug and self.apply_all_transforms:
            transformed_images = [self.basic_transform(image)]
            for t in self.transform.transforms:
                transformed_images.append(t(image))
            return transformed_images, y
        elif self.do_aug:
            X = self.transform(image)
        else:
            X = self.basic_transform(image)
        return X, y
    
def get_dataset(train=True, image_size=128, resize=False, rotate=False, normalize=False, advanced_augmentation=False, do_aug=False, apply_all_transforms=False):
    transform_list = []

    transform_list.append(transforms.ToTensor())
    if do_aug:
        if rotate:
            transform_list.append(transforms.RandomRotation(5))
        if normalize:
            transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        if advanced_augmentation:
            transform_list.append(transforms.RandomHorizontalFlip())
            transform_list.append(transforms.RandomVerticalFlip())
            transform_list.append(transforms.RandomAffine(0, translate=(0.1, 0.1)))
            transform_list.append(transforms.RandomAffine(0, shear=5))
            transform_list.append(transforms.RandomAffine(0, scale=(0.8, 1.2)))
            transform_list.append(transforms.RandomAffine((-90,90)))
            transform_list.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
            transform_list.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False))
    if resize:
        transform_list.append(transforms.Resize((image_size, image_size)))

    transform = transforms.Compose(transform_list)
    dataset = HotdogDataset(train=train, transform=transform, do_aug=do_aug, image_size=image_size, apply_all_transforms=apply_all_transforms)
    
    return dataset

 
if __name__ == "__main__":
    from rich import print
    
    apply_all_transforms = True
    do_aug = True
    rotate = True
    resize = True
    normalize = True
    advanced_augmentation = True
    

    batch_size = 64
    image_size = 128
    train_loader = get_dataset(train=True, image_size=image_size, resize=resize, rotate=rotate, normalize=normalize, 
                               advanced_augmentation=advanced_augmentation, do_aug=do_aug, apply_all_transforms = apply_all_transforms)
    test_loader = get_dataset(train=False, image_size=image_size, resize=resize, rotate=rotate, normalize=normalize, 
                               advanced_augmentation=advanced_augmentation, do_aug=do_aug, apply_all_transforms = apply_all_transforms)

    next_train = next(iter(train_loader))
    next_test = next(iter(test_loader))

    print("Length of train set: ", len(train_loader) * batch_size)
    print("Length of test set: ", len(test_loader) * batch_size)
    print("Data shape: ", next_train[0].shape)
    print("Label shape: ", next_train[1].shape)