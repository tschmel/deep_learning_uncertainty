import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=1)
        self.batch_norm2 = nn.BatchNorm2d(num_features=32)
        self.batch_norm3 = nn.BatchNorm2d(num_features=64)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.linear = nn.Linear(in_features=128 * 14 * 14, out_features=classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(self.batch_norm1(x)))
        x = nn.functional.relu(self.conv2(self.batch_norm2(x)))
        x = nn.functional.relu(self.conv3(self.batch_norm3(x)))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.linear(x)
        return x


def create_cnn_model(classes):
    model = CNN(classes)
    return model
