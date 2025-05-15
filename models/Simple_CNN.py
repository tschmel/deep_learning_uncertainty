import torch.nn as nn


class Simple_CNN(nn.Module):
    def __init__(self, feat_dim, classes):
        super(Simple_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=feat_dim, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=feat_dim)
        self.batch_norm2 = nn.BatchNorm2d(num_features=32)
        self.batch_norm3 = nn.BatchNorm2d(num_features=64)
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=classes)
        )

    def forward(self, x):
        x = nn.functional.relu(self.conv1(self.batch_norm1(x)))
        x = nn.functional.relu(self.conv2(self.batch_norm2(x)))
        x = nn.functional.relu(self.conv3(self.batch_norm3(x)))
        x = self.classification_head(x)
        return x


def create_simple_cnn_model(args):
    model = Simple_CNN(feat_dim=args.feat_dim, classes=args.classes)
    return model
