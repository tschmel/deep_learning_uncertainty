import os
import argparse
import yaml
import logging
import time
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

import load_datasets.MNIST
import load_datasets.Fashion
import load_datasets.CIFAR10
import load_datasets.FOOD101
import load_datasets.DTD

from test import test
from test import create_logger as test_logger


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()
    assert args.config is not None
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = SimpleNamespace(**config_dict)
    return config


def create_logger(args):
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    filename = 'train_' + args.dataset + '_' + args.model + '.log'
    file_handler = logging.FileHandler('./logs/' + filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def main():
    args = get_parser()
    logger = create_logger(args)
    logger.info('<<<<<<<<<<<<<<<<<<< START OF TRAINING >>>>>>>>>>>>>>>>>>>')
    start_time = time.time()
    writer = SummaryWriter(log_dir='./logs', filename_suffix=args.dataset + '_' + args.model)
    if args.model == 'simple_cnn':
        from models.Simple_CNN import create_simple_cnn_model as Model
        logger.info('Simple CNN model architecture selected')
    elif args.model == 'larger_cnn':
        from models.Larger_CNN import create_larger_cnn_model as Model
        logger.info('Larger_CNN model architecture selected')
    elif args.model == 'resnet-18':
        from models.ResNet import create_resnet_18_model as Model
        logger.info('ResNet-18 model architecture selected')
    elif args.model == 'resnet-34':
        from models.ResNet import create_resnet_34_model as Model
        logger.info('ResNet-34 model architecture selected')
    elif args.model == 'resnet-50':
        from models.ResNet import create_resnet_50_model as Model
        logger.info('ResNet-50 model architecture selected')
    elif args.model == 'resnet-101':
        from models.ResNet import create_resnet_101_model as Model
        logger.info('ResNet-101 model architecture selected')
    elif args.model == 'resnet-152':
        from models.ResNet import create_resnet_152_model as Model
        logger.info('ResNet-152 model architecture selected')
    elif args.model == 'u-net':
        from models.U_Net import create_u_net_model as Model
        logger.info('U-Net model architecture selected')
    elif args.model == 'mlp':
        from models.MLP import create_mlp_model as Model
        logger.info('MLP model architecture selected')
    elif args.model == 'skip_cnn':
        from models.Skip_CNN import create_skip_cnn_model as Model
        logger.info('Skip CNN model architecture selected')
    else:
        logger.error('Model architecture not supported!')
    model = Model(args).cuda()
    if args.resume:
        state_dict = torch.load(args.resume)
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classification_head.3")}
        model.load_state_dict(filtered_state_dict, strict=False)
        logger.info(f'Loaded model from {args.resume} to resume training. ')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.dataset == 'mnist':
        logger.info('MNIST dataset selected')
        dl_train = load_datasets.MNIST.load_mnist_train_dataset(args)
        dl_val = load_datasets.MNIST.load_mnist_test_dataset(args)
        dl_test = load_datasets.MNIST.load_mnist_test_dataset(args)
    elif args.dataset == 'fashion_mnist':
        logger.info('FashionMNIST dataset selected')
        dl_train = load_datasets.Fashion.load_fashion_mnist_train_dataset(args)
        dl_val = load_datasets.Fashion.load_fashion_mnist_test_dataset(args)
        dl_test = load_datasets.Fashion.load_fashion_mnist_test_dataset(args)
    elif args.dataset == 'cifar10':
        logger.info('CIFAR10 dataset selected')
        dl_train = load_datasets.CIFAR10.load_cifar10_train_dataset(args)
        dl_val = load_datasets.CIFAR10.load_cifar10_test_dataset(args)
        dl_test = load_datasets.CIFAR10.load_cifar10_test_dataset(args)
    elif args.dataset == 'food101':
        logger.info('FOOD101 dataset selected')
        dl_train = load_datasets.FOOD101.load_food101_train_dataset(args)
        dl_val = load_datasets.FOOD101.load_food101_test_dataset(args)
        dl_test = load_datasets.FOOD101.load_food101_test_dataset(args)
    elif args.dataset == 'dtd':
        logger.info('DTD dataset selected')
        dl_train = load_datasets.DTD.load_dtd_train_dataset(args)
        dl_val = load_datasets.DTD.load_dtd_val_dataset(args)
        dl_test = load_datasets.DTD.load_dtd_test_dataset(args)
    else:
        logger.error('Dataset not supported!')

    train(model, criterion, optimizer, writer, dl_train, dl_val, args, logger)
    save_path = './' + args.dataset + '_' + args.model + '_model_last.pth'
    torch.save(model.state_dict(), save_path)
    end_time = time.time()
    logger.info(f"Training took {end_time - start_time:.2f} seconds. ")
    logger.info('<<<<<<<<<<<<<<<<<<<< END OF TRAINING >>>>>>>>>>>>>>>>>>>>')
    test(model, dl_test, args, test_logger(args))


def train(model, loss_fn, optimizer, writer, dl_train, dl_test, args, logger):
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        acc = 0
        best_val_acc = 0

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
        logger.info(f"Epoch {epoch + 1}/{args.epochs}, Loss: {total_loss / len(dl_train):.4f}, Accuracy: {acc / len(dl_train):.4f}")
        if epoch % args.val_freq == 0:
            val_acc, val_loss = validate(model=model, dl_test=dl_test, loss_fn=loss_fn, epoch=epoch, writer=writer, args=args, logger=logger)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                save_path = './' + args.dataset + '_' + args.model + '_model_best.pth'
                torch.save(model.state_dict(), save_path)


def validate(model, dl_test, loss_fn, epoch, writer, args, logger):
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
    #logger.info('<<<<<<<<<<<<<<<<<<<< Validation Step >>>>>>>>>>>>>>>>>>>>')
    #logger.info(f"Loss: {total_loss_val / len(dl_test):.4f}, Accuracy: {total_acc_val / len(dl_test):.4f}")
    #logger.info('<<<<<<<<<<<<<<<<< End of Validation Step >>>>>>>>>>>>>>>>>')
    return total_loss_val, total_acc_val


if __name__ == '__main__':
    main()
