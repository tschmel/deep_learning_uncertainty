import argparse
import yaml
import logging
from types import SimpleNamespace
import torch
import torch.nn as nn
import numpy as np

import load_datasets.MNIST
import load_datasets.Fashion
import load_datasets.CIFAR10
import load_datasets.FOOD101
import load_datasets.DTD


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
    logger = logging.getLogger('uncertainty_logger')
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    filename = 'uncertainty_' + args.dataset + '_' + args.model + '.log'
    file_handler = logging.FileHandler('./logs/' + filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def main():
    args = get_parser()
    logger = create_logger(args)
    logger.info('<<<<<<<<<<<<<<<<<<< START OF UNCERTAINTY CALCULATION >>>>>>>>>>>>>>>>>>>')
    if args.model == 'cnn':
        from models.CNN import create_cnn_model as Model
        logger.info('CNN model architecture selected')
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
    elif args.model == 'custom':
        from models.Custom_Model import create_custom_model as Model
        logger.info('Custom model architecture selected')
    else:
        logger.error('Model architecture not supported!')
    model = Model(args).cuda()
    load_path = './' + args.dataset + '_' + args.model + '_model_best.pth'
    model.load_state_dict(torch.load(load_path, weights_only=True))
    if args.dataset == 'mnist':
        logger.info('MNIST dataset selected')
        dl_train, dl_test = load_datasets.MNIST.create_mnist_dataset(args.img_size, args.batch_size)
    elif args.dataset == 'fashion_mnist':
        logger.info('FashionMNIST dataset selected')
        dl_train, dl_test = load_datasets.Fashion.create_fashion_mnist_dataset(args.img_size, args.batch_size)
    elif args.dataset == 'cifar10':
        logger.info('CIFAR10 dataset selected')
        dl_train, dl_test = load_datasets.CIFAR10.create_cifar10_dataset(args.img_size, args.batch_size)
    elif args.dataset == 'food101':
        logger.info('FOOD101 dataset selected')
        dl_train, dl_test = load_datasets.FOOD101.create_food101_dataset(args.img_size, args.batch_size)
    elif args.dataset == 'dtd':
        logger.info('DTD dataset selected')
        dl_train, dl_test = load_datasets.DTD.create_dtd_dataset(args.img_size, args.batch_size)
    else:
        logger.error('Dataset not supported!')
    calculate_uncertainty(model, dl_test, args, logger)
    calculate_uncertainty2(model, dl_test, args, logger)
    logger.info('<<<<<<<<<<<<<<<<<<<< END OF UNCERTAINTY CALCULATION >>>>>>>>>>>>>>>>>>>>')


def calculate_uncertainty2(model, dl_test, args, logger):
    model.eval()
    logger.info('Calculating uncertainty per class 2')

    class_sums = torch.zeros(args.classes)
    class_counts = torch.zeros(args.classes)

    for inputs, targets in dl_test:
        inputs, targets = inputs.cuda(), targets.cuda()

        mc_preds = []
        for _ in range(args.uncert_epochs):
            probs = torch.softmax(model(inputs), dim=1)
            mc_preds.append(probs.unsqueeze(0).detach().cpu())

        mc_preds = np.concatenate(mc_preds, axis=0)
        var_per_class = mc_preds.var(axis=0)

        for i in range(inputs.size(0)):  # loop over batch
            label = targets[i].item()
            class_sums[label] += var_per_class[i, label]  # uncertainty for true class
            class_counts[label] += 1

    class_conditional_uncertainty = class_sums / class_counts
    logger.info(f"Average uncertainty per class: {list(map('{:.4f}'.format, class_conditional_uncertainty))}")


def calculate_uncertainty(model, dl_test, args, logger):
    model.eval()
    logger.info('Calculating uncertainty per class')
    for layer in model.modules():
        if isinstance(layer, nn.Dropout):
            layer.train()

    total_var = np.zeros(args.classes)
    total_count = 0

    for inputs, _ in dl_test:
        inputs = inputs.cuda()

        mc_preds = []
        for _ in range(args.uncert_epochs):
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            mc_preds.append(probs.unsqueeze(0).detach().cpu())

        mc_preds = np.concatenate(mc_preds, axis=0)
        var_per_class = mc_preds.var(axis=0)

        total_var += var_per_class.sum(axis=0)
        total_count += inputs.size(0)

    avg_class_uncertainty = total_var / total_count
    logger.info(f"Average uncertainty per class: {list(map('{:.4f}'.format, avg_class_uncertainty))}")


if __name__ == '__main__':
    main()
