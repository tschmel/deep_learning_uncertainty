import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def cifar10(args, split):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dl = None
    if split == 'train':
        train_dataset = torchvision.datasets.CIFAR10(root='../datasets', train=True, transform=train_transform, download=True)
        dl = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    elif split == 'test':
        test_dataset = torchvision.datasets.CIFAR10(root='../datasets', train=False, transform=test_transform, download=True)
        dl = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    return dl


def load_cifar10_train_dataset(args):
    return cifar10(args, split='train')


def load_cifar10_test_dataset(args):
    return cifar10(args, split='test')

