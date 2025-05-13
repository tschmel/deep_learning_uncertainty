import torch.nn as nn


class Larger_CNN(nn.Module):
    def __init__(self, feat_dim, classes):
        super(Larger_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=feat_dim, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(num_features=feat_dim)
        self.batch_norm2 = nn.BatchNorm2d(num_features=16)
        self.batch_norm3 = nn.BatchNorm2d(num_features=32)
        self.batch_norm4 = nn.BatchNorm2d(num_features=64)
        self.batch_norm5 = nn.BatchNorm2d(num_features=128)
        self.batch_norm6 = nn.BatchNorm2d(num_features=256)
        self.batch_norm7 = nn.BatchNorm2d(num_features=512)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.3)
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=classes)
        )

    def forward(self, x):
        x = nn.functional.relu(self.conv1(self.batch_norm1(x)))
        x = self.pool1(nn.functional.relu(self.conv2(self.batch_norm2(x))))
        x = self.pool2(nn.functional.relu(self.conv3(self.batch_norm3(x))))
        x = self.pool3(nn.functional.relu(self.conv4(self.batch_norm4(x))))
        x = self.pool4(nn.functional.relu(self.conv5(self.batch_norm5(x))))
        x = self.pool5(nn.functional.relu(self.conv6(self.batch_norm6(x))))
        x = self.dropout1(nn.functional.relu(self.conv7(self.batch_norm7(x))))
        x = self.classification_head(x)
        return x


def create_larger_cnn_model(args):
    model = Larger_CNN(feat_dim=args.feat_dim, classes=args.classes)
    return model
