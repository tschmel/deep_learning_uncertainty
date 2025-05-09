import argparse
import yaml
import logging
from types import SimpleNamespace
import torch
import torchmetrics

import load_datasets.MNIST
import load_datasets.CIFAR10
import load_datasets.FOOD101


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
    logger = logging.getLogger('testing_logger')
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    filename = 'test_' + args.dataset + '_' + args.model + '.log'
    file_handler = logging.FileHandler('./logs/' + filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def main():
    args = get_parser()
    logger = create_logger(args)
    logger.info('<<<<<<<<<<<<<<<<<<< START OF TESTING >>>>>>>>>>>>>>>>>>>')
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
    elif args.model == 'mlp':
        from models.MLP import create_mlp_model as Model
        logger.info('MLP model architecture selected')
    else:
        logger.error('Model architecture not supported!')
    model = Model(args).cuda()
    load_path = './' + args.dataset + '_' + args.model + '_model_best.pth'
    model.load_state_dict(torch.load(load_path, weights_only=True))
    if args.dataset == 'mnist':
        logger.info('MNIST dataset selected')
        dl_train, dl_test = load_datasets.MNIST.create_mnist_dataset(args.img_size, args.batch_size)
    elif args.dataset == 'cifar10':
        logger.info('CIFAR10 dataset selected')
        dl_train, dl_test = load_datasets.CIFAR10.create_cifar10_dataset(args.img_size, args.batch_size)
    elif args.dataset == 'food101':
        logger.info('FOOD101 dataset selected')
        dl_train, dl_test = load_datasets.FOOD101.create_food101_dataset(args.img_size, args.batch_size)
    else:
        logger.error('Dataset not supported!')
    test(model, dl_test, args, logger)
    logger.info('<<<<<<<<<<<<<<<<<<<< END OF TESTING >>>>>>>>>>>>>>>>>>>>')


def test(model, dl_test, args, logger):
    model.eval()
    logger.info('Start evaluating model...')
    total_acc = 0
    with torch.no_grad():
        for data, labels in dl_test:
            data, labels = data.cuda(), labels.cuda()
            outputs = model(data)
            acc = torchmetrics.functional.accuracy(outputs, labels, task='multiclass', num_classes=args.classes)
            total_acc += acc

    total_acc = (total_acc/len(dl_test))
    logger.info(f"Accuracy {total_acc:.4f}")


if __name__ == '__main__':
    main()
