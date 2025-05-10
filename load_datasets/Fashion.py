import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def create_fashion_mnist_dataset(img_size, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),  # original: 28x28
    ])

    # Load dataset
    train_dataset = torchvision.datasets.FashionMNIST(root='../datasets', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.FashionMNIST(root='../datasets', train=False, transform=transform, download=True)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
