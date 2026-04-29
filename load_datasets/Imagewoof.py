from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

def load_imagewoof(args, split):
    root = '../datasets/imagewoof/imagewoof2-160'

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
        train_dataset = ImageFolder(root=f'{root}/train', transform=train_transform)
        dl = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, pin_memory=True)
    elif split == 'val':
        val_dataset = ImageFolder(root=f'{root}/val', transform=test_transform)
        dl = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)
    return dl


def load_imagewoof_train_dataset(args):
    return load_imagewoof(args, split='train')


def load_imagewoof_val_dataset(args):
    return load_imagewoof(args, split='val')


def load_imagewoof_test_dataset(args):
    return load_imagewoof(args, split='val')