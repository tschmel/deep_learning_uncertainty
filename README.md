# Uncertainty quantification classification models
This repository is a personal project aimed at exploring and understanding different deep learning model architectures and the concept of uncertainty quantification in predictive models. \
Different model architectures have their own advantages and disadvantages. 

## Overview
With this repository you can: 
- Train and evaluate various deep learning architectures
- Use multiple image datasets from the ```torchvision``` library 
- Apply Monte Carlo Dropout for uncertainty quantification 

## Implemented Architectures
- MLP (Multi-Layer Perceptron)
- Simple CNN
- Larger CNN
- Skip CNN 
- ResNet Variants: ResNet-19, ResNet-34, ResNet-50, ResNet-101, ResNet-152
- U-Net (commonly used for segmentation tasks)

## Supported Datasets
The following datasets from ```torchvision.datasets``` are supported:
- MNIST
- FashionMNIST
- CIFAR-10
- FOOD101
- DTD (Describable Textures Dataset)

## Uncertainty Quantification 
Modern deep learning models are powerful, but they often output confident predictions even when they are wrong. \
Uncertainty quantification helps address this by measuring how "sure" a model is about its predictions. \
This is particularly important when:
- Making decisions in critical applications (e.g., medical imaging, autonomous driving)
- Working with noisy or out-of-distribution data 
- Combining predictions from multiple models or datasets 

### Monte Carlo Dropout (MC Dropout) 
Monte Carlo Dropout is a practical method to estimate uncertainty by performing multiple forward passes with dropout enabled at test time and analyzing the distribution of the outputs. \
Normally, dropout is used during training to prevent overfitting by randomly turning off neurons in the network. During inference, dropout is typically disabled to use the full model capacity. However, in MC Dropout, dropout remains active even during inference. \
By performing multiple stochastic forward passes through the model (each with different neurons dropped), we can: 
- Collect multiple predictions for the same input 
- Estimate the mean prediction and the variance (uncertainty) across the predictions

MC Dropout can be interpreted as an approximation to Bayesian inference in neural networks. While full Bayesian deep learning is often computationally expensive, MC Dropout provides a practical and scalable way to get uncertainty estimates using standard neural networks.

## How to use this repository 
In order to train, test quantify uncertainty, you have to use a config file that provides different parameters. \
There are already some available in the ```./logs``` directory. \
First you have to install the necessary packages via: \
```pip install -r requirements.txt``` 

To train a model you run: \
```python3 train.py --config <path_to_config_yaml>``` 

To test a model you run: \
```python3 test.py --config <path_to_config_yaml>``` 

To do uncertainty quantification you run: \
```python3 calculate_uncertainty.py --config <path_to_config_yaml>``` 


## Goals of This Project
This project was created to: 
- Learn how to build and implement various deep learning architectures from scratch 
- Experiment with different types of computer vision datasets 
- Explore how uncertainty can be quantified and interpreted in neural networks 