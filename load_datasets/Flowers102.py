import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def flowers102(args, split):
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # original: 256x256 (?)
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),  # original: 256x256 (?)
    ])
    dl = None
    if split == 'train':
        train_dataset = torchvision.datasets.Flowers102(root='../datasets', split='train', transform=test_transform, download=True)
        dl = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    elif split == 'val':
        val_dataset = torchvision.datasets.Flowers102(root='../datasets', split='val', transform=test_transform, download=True)
        dl = DataLoader(val_dataset, batch_size=args.train_batch_size, shuffle=False)
    elif split == 'test':
        test_dataset = torchvision.datasets.Flowers102(root='../datasets', split='test', transform=test_transform, download=True)
        dl = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    return dl


def load_flowers102_train_dataset(args):
    return flowers102(args, split='train')


def load_flowers102_val_dataset(args):
    return flowers102(args, split='val')


def load_flowers102_test_dataset(args):
    return flowers102(args, split='test')
