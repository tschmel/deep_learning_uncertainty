import argparse
import yaml
from types import SimpleNamespace
import torch
import torch.nn as nn
import numpy as np

import load_datasets.MNIST
import models.CNN
from models import *
from load_datasets import *


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()
    assert args.config is not None
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = SimpleNamespace(**config_dict)
    return config


def main():
    args = get_parser()
    model = models.CNN.create_cnn_model(classes=args.classes).cuda()
    model.load_state_dict(torch.load('./model.pth', weights_only=True))
    dl_train, dl_test = load_datasets.MNIST.create_mnist_dataset(args.batch_size)
    calculate_uncertainty(model, dl_test, args)


def calculate_uncertainty(model, dl_test, args):
    model.eval()
    for layer in model.modules():
        if isinstance(layer, nn.Dropout2d):
            layer.train()

    total_var = np.zeros(args.classes)
    total_count = 0

    for inputs, _ in dl_test:
        inputs = inputs.cuda()

        mc_preds = []
        for _ in range(args.uncert_epochs):
            probs = torch.softmax(model(inputs), dim=1)
            mc_preds.append(probs.unsqueeze(0).detach().cpu())

        mc_preds = np.concatenate(mc_preds, axis=0)     # shape: (T, B, N)
        var_per_class = mc_preds.var(axis=0)       # shape: (B, N)

        total_var += var_per_class.sum(axis=0)     # sum over batch, keep per-class
        total_count += inputs.size(0)             # total number of samples

    avg_class_uncertainty = total_var / total_count
    print(avg_class_uncertainty)


if __name__ == '__main__':
    main()
