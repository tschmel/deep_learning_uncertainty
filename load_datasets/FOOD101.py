import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def food101(args, split):
    train_transform = transforms.Compose([
        transforms.Resize(256),  # original: max 512x512
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),  # original: max 512x512
        transforms.ToTensor(),
    ])
    dl = None
    if split == 'train':
        train_dataset = torchvision.datasets.Food101(root='../datasets', split='train', transform=train_transform, download=True)
        dl = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    elif split == 'test':
        test_dataset = torchvision.datasets.Food101(root='../datasets', split='test', transform=test_transform, download=True)
        dl = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    return dl


def load_food101_train_dataset(args):
    return food101(args, split='train')


def load_food101_test_dataset(args):
    return food101(args, split='test')
