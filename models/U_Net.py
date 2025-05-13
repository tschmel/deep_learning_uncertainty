import torch
import torch.nn as nn


class U_Net_Block(nn.Module):
    def __init__(self, in_feat_dim, out_feat_dim):
        super(U_Net_Block, self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(num_features=in_feat_dim)
        self.conv1 = nn.Conv2d(in_channels=in_feat_dim, out_channels=out_feat_dim, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_feat_dim)
        self.conv2 = nn.Conv2d(in_channels=out_feat_dim, out_channels=out_feat_dim, kernel_size=3, stride=1, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(num_features=out_feat_dim)
        self.conv3 = nn.Conv2d(in_channels=out_feat_dim, out_channels=out_feat_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(nn.functional.relu(self.batch_norm1(x)))
        x = self.conv2(nn.functional.relu(self.batch_norm2(x)))
        x = self.conv3(nn.functional.relu(self.batch_norm3(x)))
        return x


class U_Net(nn.Module):
    def __init__(self, feat_dim, classes):
        super(U_Net, self).__init__()
        self.downsample_block1 = U_Net_Block(feat_dim, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.downsample_block2 = U_Net_Block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.downsample_block3 = U_Net_Block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.downsample_block4 = U_Net_Block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.downsample_block5 = U_Net_Block(512, 1024)

        self.upsample1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0)
        self.upsample_block1 = U_Net_Block(1024, 512)
        self.upsample2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0)
        self.upsample_block2 = U_Net_Block(512, 256)
        self.upsample3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0)
        self.upsample_block3 = U_Net_Block(256, 128)
        self.upsample4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.upsample_block4 = U_Net_Block(128, 64)

        self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=64, out_features=classes)
            )

    def forward(self, x):
        x = self.downsample_block1(x)
        skip1 = x
        x = self.pool1(x)
        x = self.downsample_block2(x)
        skip2 = x
        x = self.pool2(x)
        x = self.downsample_block3(x)
        skip3 = x
        x = self.pool3(x)
        x = self.downsample_block4(x)
        skip4 = x
        x = self.pool4(x)
        x = self.downsample_block5(x)

        x = self.upsample1(x)
        x = torch.cat((skip4, x), dim=1)
        x = self.upsample_block1(x)
        x = self.upsample2(x)
        x = torch.cat((skip3, x), dim=1)
        x = self.upsample_block2(x)
        x = self.upsample3(x)
        x = torch.cat((skip2, x), dim=1)
        x = self.upsample_block3(x)
        x = self.upsample4(x)
        x = torch.cat((skip1, x), dim=1)
        x = self.upsample_block4(x)

        x = self.classification_head(x)
        return x


def create_u_net_model(args):
    model = U_Net(feat_dim=args.feat_dim, classes=args.classes)
    return model
