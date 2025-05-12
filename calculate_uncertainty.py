import argparse
import yaml
import logging
import sys
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
    load_path = args.test_path
    model.load_state_dict(torch.load(load_path, weights_only=True))
    if args.dataset == 'mnist':
        logger.info('MNIST dataset selected')
        dl_test = load_datasets.MNIST.load_mnist_test_dataset(args)
    elif args.dataset == 'fashion_mnist':
        logger.info('FashionMNIST dataset selected')
        dl_test = load_datasets.Fashion.load_fashion_mnist_test_dataset(args)
    elif args.dataset == 'cifar10':
        dl_test = load_datasets.CIFAR10.load_cifar10_test_dataset(args)
    elif args.dataset == 'food101':
        logger.info('FOOD101 dataset selected')
        dl_test = load_datasets.FOOD101.load_food101_test_dataset(args)
    elif args.dataset == 'dtd':
        logger.info('DTD dataset selected')
        dl_test = load_datasets.DTD.load_dtd_test_dataset(args)
    else:
        logger.error('Dataset not supported!')
    calculate_uncertainty(model, dl_test, args, logger)
    logger.info('<<<<<<<<<<<<<<<<<<<< END OF UNCERTAINTY CALCULATION >>>>>>>>>>>>>>>>>>>>')


def calculate_uncertainty(model, dl_test, args, logger):
    model.eval()
    logger.info('Calculating uncertainty ')

    # Storage
    all_entropies = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dl_test:
            inputs = inputs.cuda()
            labels = labels.cuda()

            # Collect MC predictions
            probs_mc = []
            for _ in range(args.uncert_epochs):
                logits = model(inputs)
                probs = nn.functional.softmax(logits, dim=1)
                probs_mc.append(probs.unsqueeze(0).detach().cpu().numpy())

            # Stack and compute mean
            probs_mc = np.concatenate(probs_mc, axis=0)
            mean_probs = np.mean(probs_mc, axis=0)

            # Compute predictive entropy per sample
            # H = - sum_k p_k * log(p_k)
            entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-12), axis=1)

            all_entropies.append(entropy)
            all_labels.append(labels.cpu().numpy())

    # Concatenate
    all_entropies = np.concatenate(all_entropies, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Aggregate by class
    classes = np.unique(all_labels)
    class_uncertainty = {}
    for cls in classes:
        mask = (all_labels == cls)
        class_uncertainty[int(cls)] = float(all_entropies[mask].mean())

    # Overall
    overall_uncertainty = float(all_entropies.mean())

    logger.info('Uncertainty per class: ')
    for cls, unc in class_uncertainty.items():
        logger.info(f' Class {cls}: {unc:.4f}')
    logger.info(f'Overall uncertainty: {overall_uncertainty:.4f}')


if __name__ == '__main__':
    main()
