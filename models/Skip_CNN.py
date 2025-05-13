import torch.nn as nn


class Skip_CNN_Block(nn.Module):
    def __init__(self, in_feat_dim, out_feat_dim):
        super(Skip_CNN_Block, self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(in_feat_dim)
        self.conv1 = nn.Conv2d(in_channels=in_feat_dim, out_channels=out_feat_dim, kernel_size=3, stride=2, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_feat_dim)
        self.conv2 = nn.Conv2d(in_channels=out_feat_dim, out_channels=out_feat_dim, kernel_size=3, stride=1, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(in_feat_dim)
        self.conv3 = nn.Conv2d(in_channels=in_feat_dim, out_channels=out_feat_dim, kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        skip = x
        x = self.conv1(nn.functional.relu(self.batch_norm1(x)))
        x = self.conv2(nn.functional.relu(self.batch_norm2(x)))
        skip = self.conv3(nn.functional.relu(self.batch_norm3(skip)))
        x += skip
        return x


class Skip_CNN(nn.Module):
    def __init__(self, feat_dim, classes):
        super(Skip_CNN, self).__init__()
        self.extractor = nn.Sequential(
            Skip_CNN_Block(in_feat_dim=feat_dim, out_feat_dim=32),
            Skip_CNN_Block(in_feat_dim=32, out_feat_dim=64),
            Skip_CNN_Block(in_feat_dim=64, out_feat_dim=128)
        )
        self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(p=0.4),
                nn.Linear(in_features=128, out_features=classes)
            )

    def forward(self, x):
        x = self.extractor(x)
        x = self.classification_head(x)
        return x


def create_skip_cnn_model(args):
    model = Skip_CNN(feat_dim=args.feat_dim, classes=args.classes)
    return model
