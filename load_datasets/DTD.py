import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def dtd(args, split):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.img_size, args.img_size)),  # original: 28x28
    ])
    dl = None
    if split == 'train':
        train_dataset = torchvision.datasets.DTD(root='../datasets', split='train', transform=transform, download=True)
        dl = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    elif split == 'val':
        val_dataset = torchvision.datasets.Food101(root='../datasets', split='val', transform=transform, download=True)
        dl = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False)
    elif split == 'test':
        test_dataset = torchvision.datasets.DTD(root='../datasets', split='test', transform=transform, download=True)
        dl = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    return dl


def load_dtd_train_dataset(args):
    return dtd(args, split='train')


def load_dtd_val_dataset(args):
    return dtd(args, split='val')


def load_dtd_test_dataset(args):
    return dtd(args, split='test')
