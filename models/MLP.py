import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, img_size, feat_dim, classes):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=img_size * img_size * feat_dim, out_features=1024)
        self.linear2 = nn.Linear(in_features=1024, out_features=2048)
        self.linear3 = nn.Linear(in_features=2048, out_features=2048)
        self.batch_norm1 = nn.BatchNorm1d(num_features=img_size * img_size * feat_dim)
        self.batch_norm2 = nn.BatchNorm1d(num_features=1024)
        self.batch_norm3 = nn.BatchNorm1d(num_features=2048)
        self.dropout1 = nn.Dropout(p=0.5)
        self.classification_head = nn.Linear(in_features=2048, out_features=classes)

    def forward(self, x):
        x = self.flatten(x)
        x = nn.functional.relu(self.linear1(self.batch_norm1(x)))
        x = nn.functional.relu(self.linear2(self.batch_norm2(x)))
        x = nn.functional.relu(self.linear3(self.batch_norm3(x)))
        x = self.dropout1(x)
        x = self.classification_head(x)
        return x


def create_mlp_model(args):
    model = MLP(img_size=args.img_size, feat_dim = args.feat_dim, classes=args.classes)
    return model
