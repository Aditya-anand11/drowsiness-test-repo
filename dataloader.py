import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms

class FolderImageDataset(Dataset):
    def __init__(self, class0_dir, class1_dir, transform=None):
        self.transform = transform
        self.samples = []
        for path in glob.glob(os.path.join(class0_dir, "*.*")):
            self.samples.append((path, 0))
        for path in glob.glob(os.path.join(class1_dir, "*.*")):
            self.samples.append((path, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def get_transforms(img_size=(64, 64), augment=True):
    if augment:
        train_tf = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor()
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])
    test_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    return train_tf, test_tf

def get_loaders(dataset_type="both", img_size=(64, 64), batch_size=32, augment=True):
    train_tf, test_tf = get_transforms(img_size, augment)

    eye_train = datasets.ImageFolder("train", transform=train_tf)
    eye_test = datasets.ImageFolder("test", transform=test_tf)

    yawn_full = FolderImageDataset("no yawn", "yawn", transform=train_tf)
    yawn_train_len = int(0.8 * len(yawn_full))
    yawn_test_len = len(yawn_full) - yawn_train_len
    yawn_train, yawn_test = random_split(yawn_full, [yawn_train_len, yawn_test_len])

    if dataset_type == "eyes":
        train_set, test_set = eye_train, eye_test
    elif dataset_type == "yawn":
        train_set, test_set = yawn_train, yawn_test
    else:
        train_set = ConcatDataset([eye_train, yawn_train])
        test_set = ConcatDataset([eye_test, yawn_test])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

class SequenceDataset(Dataset):
    def __init__(self, num_samples=500, seq_len=10, feature_dim=4):
        self.X = torch.randn(num_samples, seq_len, feature_dim)
        self.y = (self.X[:, :, 0].mean(dim=1) > 0).float().unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_sequence_loader(batch_size=32):
    dataset = SequenceDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
