import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def load_imagenette(args, split):
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    dl = None
    if split == 'train':
        train_dataset = torchvision.datasets.Imagenette(root='../datasets/imagenette', split='train', size='160px',
                                                        transform=train_transform, download=True)
        dl = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    elif split == 'val':
        test_dataset = torchvision.datasets.Imagenette(root='../datasets/imagenette', split='val', size='160px',
                                                      transform=test_transform, download=True)
        dl = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    return dl


def load_imagenette_train_dataset(args):
    return load_imagenette(args, split='train')


def load_imagenette_val_dataset(args):
    return load_imagenette(args, split='val')


def load_imagenette_test_dataset(args):
    return load_imagenette(args, split='val')
