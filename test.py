import argparse
import yaml
from types import SimpleNamespace
import torch
import torchmetrics

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
    test(model, dl_test, args)


def test(model, dl_test, args):
    model.eval()
    total_acc = 0
    with torch.no_grad():
        for data, labels in dl_test:
            data, labels = data.cuda(), labels.cuda()
            outputs = model(data)
            acc = torchmetrics.functional.accuracy(outputs, labels, task='multiclass', num_classes=args.classes)
            total_acc += acc

    total_acc = (total_acc/len(dl_test))
    print('Accuracy: ', total_acc)


if __name__ == '__main__':
    main()
