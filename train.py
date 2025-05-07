import argparse
import yaml
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
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
    writer = SummaryWriter()
    model = models.CNN.create_cnn_model(classes=args.classes).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    dl_train, dl_test = load_datasets.MNIST.create_mnist_dataset(args.batch_size)
    train(model, criterion, optimizer, writer, dl_train, dl_test, args)
    torch.save(model.state_dict(), './model.pth')


def train(model, loss_fn, optimizer, writer, dl_train, dl_test, args):
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        acc = 0

        for images, labels in dl_train:
            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            acc += torchmetrics.functional.accuracy(outputs, labels, task='multiclass', num_classes=args.classes)

        writer.add_scalar('loss/train', total_loss / len(dl_train), epoch)
        writer.add_scalar('acc/train', acc / len(dl_train), epoch)
        if (epoch + 1) % args.val_freq == 0:
            validate(model=model, dl_test=dl_test, loss_fn=loss_fn, epoch=epoch, writer=writer, args=args)

        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {total_loss / len(dl_train):.4f}, Accuracy: {acc / len(dl_train):.4f}")


def validate(model, dl_test, loss_fn, epoch, writer, args):
    model.eval()
    total_loss_val = 0
    total_acc_val = 0
    with torch.no_grad():
        for data, labels in dl_test:
            data, labels = data.cuda(), labels.cuda()
            outputs = model(data)
            loss = loss_fn(outputs, labels)
            total_loss_val += loss.item()
            acc = torchmetrics.functional.accuracy(outputs, labels, task='multiclass', num_classes=args.classes)
            total_acc_val += acc

    writer.add_scalar('loss/val', total_loss_val / len(dl_test), epoch)
    writer.add_scalar('acc/val', total_acc_val / len(dl_test), epoch)


if __name__ == '__main__':
    main()
