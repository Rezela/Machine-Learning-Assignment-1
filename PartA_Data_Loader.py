import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(batch_size=64):
    # prepare
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2470, 0.2435, 0.2616])
    ])

    # download
    train_set = torchvision.datasets.CIFAR10(root='./data',train=True, download=True,transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data',train=False, download=True, transform=transform)

    # divide into validation set and test set
    val_size = len(test_set)//2
    test_size = len(test_set) - val_size
    val_set, test_set = random_split(test_set, [val_size, test_size])

    # create data loaders
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader

