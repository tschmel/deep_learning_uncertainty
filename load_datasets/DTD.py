import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def create_dtd_dataset(img_size, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),  # original: 300x300 - 600x600
    ])

    # Load dataset
    train_dataset = torchvision.datasets.DTD(root='../datasets', split='train', transform=transform, download=True)
    test_dataset = torchvision.datasets.DTD(root='../datasets', split='test', transform=transform, download=True)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
