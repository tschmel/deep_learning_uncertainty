import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def fashion_mnist(args, split):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.img_size, args.img_size)),  # original: 28x28
    ])
    dl = None
    if split == 'train':
        train_dataset = torchvision.datasets.FashionMNIST(root='../datasets', train=True, transform=transform, download=True)
        dl = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    elif split == 'test':
        test_dataset = torchvision.datasets.FashionMNIST(root='../datasets', train=False, transform=transform, download=True)
        dl = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    return dl


def load_fashion_mnist_train_dataset(args):
    return fashion_mnist(args, split='train')


def load_fashion_mnist_test_dataset(args):
    return fashion_mnist(args, split='test')
