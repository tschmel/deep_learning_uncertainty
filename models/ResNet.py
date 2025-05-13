import torch.nn as nn


class Stage1_BottleNeckBlock(nn.Module):
    def __init__(self, in_feat_dim, out_feat_dim):
        super(Stage1_BottleNeckBlock, self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(num_features=in_feat_dim)
        self.conv1 = nn.Conv2d(in_channels=in_feat_dim, out_channels=in_feat_dim, kernel_size=1, stride=1, padding=0)
        self.batch_norm2 = nn.BatchNorm2d(num_features=in_feat_dim)
        self.conv2 = nn.Conv2d(in_channels=in_feat_dim, out_channels=in_feat_dim, kernel_size=3, stride=1, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(num_features=in_feat_dim)
        self.conv3 = nn.Conv2d(in_channels=in_feat_dim, out_channels=out_feat_dim, kernel_size=1, stride=1, padding=0)
        self.batch_norm4 = nn.BatchNorm2d(num_features=in_feat_dim)
        self.conv4 = nn.Conv2d(in_channels=in_feat_dim, out_channels=out_feat_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        skip = x
        x = self.conv1(nn.functional.relu(self.batch_norm1(x)))
        x = self.conv2(nn.functional.relu(self.batch_norm2(x)))
        x = self.conv3(nn.functional.relu(self.batch_norm3(x)))
        skip = self.conv4(nn.functional.relu(self.batch_norm4(skip)))
        x = x + skip
        return x


class DownSampleBootleNeckBlock(nn.Module):
    def __init__(self, in_feat_dim, out_feat_dim):
        super(DownSampleBootleNeckBlock, self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(num_features=in_feat_dim)
        self.conv1 = nn.Conv2d(in_channels=in_feat_dim, out_channels=int(in_feat_dim / 2), kernel_size=1, stride=1, padding=0)
        self.batch_norm2 = nn.BatchNorm2d(num_features=int(in_feat_dim / 2))
        self.conv2 = nn.Conv2d(in_channels=int(in_feat_dim / 2), out_channels=int(in_feat_dim / 2), kernel_size=3, stride=2, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(num_features=int(in_feat_dim / 2))
        self.conv3 = nn.Conv2d(in_channels=int(in_feat_dim / 2), out_channels=out_feat_dim, kernel_size=1, stride=1, padding=0)
        self.batch_norm4 = nn.BatchNorm2d(num_features=in_feat_dim)
        self.conv4 = nn.Conv2d(in_channels=in_feat_dim, out_channels=out_feat_dim, kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        skip = x
        x = self.conv1(nn.functional.relu(self.batch_norm1(x)))
        x = self.conv2(nn.functional.relu(self.batch_norm2(x)))
        x = self.conv3(nn.functional.relu(self.batch_norm3(x)))
        skip = self.conv4(nn.functional.relu(self.batch_norm4(skip)))
        x = x + skip
        return x


class BottleNeckBlock(nn.Module):
    def __init__(self, feat_dim):
        super(BottleNeckBlock, self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(num_features=feat_dim)
        self.conv1 = nn.Conv2d(in_channels=feat_dim, out_channels=int(feat_dim / 4), kernel_size=1, stride=1, padding=0)
        self.batch_norm2 = nn.BatchNorm2d(num_features=int(feat_dim / 4))
        self.conv2 = nn.Conv2d(in_channels=int(feat_dim / 4), out_channels=int(feat_dim / 4), kernel_size=3, stride=1, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(num_features=int(feat_dim / 4))
        self.conv3 = nn.Conv2d(in_channels=int(feat_dim / 4), out_channels=feat_dim, kernel_size=1, stride=1, padding=0)
        self.batch_norm4 = nn.BatchNorm2d(num_features=feat_dim)
        self.conv4 = nn.Conv2d(in_channels=feat_dim, out_channels=feat_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        skip = x
        x = self.conv1(nn.functional.relu(self.batch_norm1(x)))
        x = self.conv2(nn.functional.relu(self.batch_norm2(x)))
        x = self.conv3(nn.functional.relu(self.batch_norm3(x)))
        skip = self.conv4(nn.functional.relu(self.batch_norm4(skip)))
        x = x + skip
        return x


class DownSampleResNetBlock(nn.Module):
    def __init__(self, in_feat_dim, out_feat_dim):
        super(DownSampleResNetBlock, self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(num_features=in_feat_dim)
        self.conv1 = nn.Conv2d(in_channels=in_feat_dim, out_channels=out_feat_dim, kernel_size=3, stride=2, padding=0)
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_feat_dim)
        self.conv2 = nn.Conv2d(in_channels=out_feat_dim, out_channels=out_feat_dim, kernel_size=3, stride=1, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(num_features=in_feat_dim)
        self.conv3 = nn.Conv2d(in_channels=in_feat_dim, out_channels=out_feat_dim, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        skip = x
        x = self.conv1(nn.functional.relu(self.batch_norm1(x)))
        x = self.conv2(nn.functional.relu(self.batch_norm2(x)))
        skip = self.conv3(nn.functional.relu(self.batch_norm3(skip)))
        x = x + skip
        return x


class ResNetBlock(nn.Module):
    def __init__(self, feat_dim):
        super(ResNetBlock, self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(num_features=feat_dim)
        self.conv1 = nn.Conv2d(in_channels=feat_dim, out_channels=feat_dim, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(num_features=feat_dim)
        self.conv2 = nn.Conv2d(in_channels=feat_dim, out_channels=feat_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        skip = x
        x = self.conv1(nn.functional.relu(self.batch_norm1(x)))
        x = self.conv2(nn.functional.relu(self.batch_norm2(x)))
        x = x + skip
        return x


class Stage(nn.Module):
    def __init__(self, num_blocks, in_feat_dim, out_feat_dim, bottle_neck):
        super(Stage, self).__init__()
        if bottle_neck:
            if in_feat_dim == out_feat_dim / 2:
                modules = [DownSampleBootleNeckBlock(in_feat_dim=in_feat_dim, out_feat_dim=out_feat_dim)]
            else:
                modules = [Stage1_BottleNeckBlock(in_feat_dim=in_feat_dim, out_feat_dim=out_feat_dim)]
            for _ in range(num_blocks - 1):
                modules.append(BottleNeckBlock(feat_dim=out_feat_dim))
        else:
            if in_feat_dim != out_feat_dim:
                modules = [DownSampleResNetBlock(in_feat_dim=in_feat_dim, out_feat_dim=out_feat_dim)]
            else:
                modules = [ResNetBlock(feat_dim=in_feat_dim)]
            for _ in range(num_blocks - 1):
                modules.append(ResNetBlock(feat_dim=out_feat_dim))
        self.stage = nn.Sequential(*modules)

    def forward(self, x):
        return self.stage(x)


class ResNet(nn.Module):
    def __init__(self, num_blocks_per_stage, bottle_neck, feat_dim, classes):
        super(ResNet, self).__init__()
        self.input_stem = nn.Sequential(
            nn.BatchNorm2d(num_features=feat_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=feat_dim, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        if bottle_neck:
            self.stage1 = Stage(num_blocks=num_blocks_per_stage[0], in_feat_dim=64, out_feat_dim=256, bottle_neck=True)
            self.stage2 = Stage(num_blocks=num_blocks_per_stage[1], in_feat_dim=256, out_feat_dim=512, bottle_neck=True)
            self.stage3 = Stage(num_blocks=num_blocks_per_stage[2], in_feat_dim=512, out_feat_dim=1024, bottle_neck=True)
            self.stage4 = Stage(num_blocks=num_blocks_per_stage[3], in_feat_dim=1024, out_feat_dim=2048, bottle_neck=True)
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=2048, out_features=classes)
            )
        else:
            self.stage1 = Stage(num_blocks=num_blocks_per_stage[0], in_feat_dim=64, out_feat_dim=64, bottle_neck=False)
            self.stage2 = Stage(num_blocks=num_blocks_per_stage[1], in_feat_dim=64, out_feat_dim=128, bottle_neck=False)
            self.stage3 = Stage(num_blocks=num_blocks_per_stage[2], in_feat_dim=128, out_feat_dim=256, bottle_neck=False)
            self.stage4 = Stage(num_blocks=num_blocks_per_stage[3], in_feat_dim=256, out_feat_dim=512, bottle_neck=False)
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=512, out_features=classes)
            )

    def forward(self, x):
        x = self.input_stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.classification_head(x)
        return x


def create_resnet_18_model(args):
    model = ResNet(num_blocks_per_stage=[2, 2, 2, 2], bottle_neck=False, feat_dim=args.feat_dim, classes=args.classes)
    return model


def create_resnet_34_model(args):
    model = ResNet(num_blocks_per_stage=[3, 4, 6, 3], bottle_neck=False, feat_dim=args.feat_dim, classes=args.classes)
    return model


def create_resnet_50_model(args):
    model = ResNet(num_blocks_per_stage=[3, 4, 6, 3], bottle_neck=True, feat_dim=args.feat_dim, classes=args.classes)
    return model


def create_resnet_101_model(args):
    model = ResNet(num_blocks_per_stage=[3, 4, 23, 3], bottle_neck=True, feat_dim=args.feat_dim, classes=args.classes)
    return model


def create_resnet_152_model(args):
    model = ResNet(num_blocks_per_stage=[3, 8, 36, 3], bottle_neck=True, feat_dim=args.feat_dim, classes=args.classes)
    return model
