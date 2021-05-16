# Deeper-Networks-for-Image-Classification

By Xu Dong

Email: xudong9771@gmail.com

This repository provides three different deeper network on image classification tasks: VGG, ResNet, DenseNet on MNIST and CIFAR-10 datasets.

In this work we perform and evaluate the image classification tasks in terms of model configuration comparison, training, validation and testing losses and error rate.

We also provide improved methods on all models by using data augmentation, Dropout and batch-normalization.

Our results are shown as below:

## Model architecture
![paper](https://user-images.githubusercontent.com/33721483/118407401-81815680-b678-11eb-9723-8e596287a2d0.png)

## Training on vanilla method
![image](https://user-images.githubusercontent.com/33721483/118407330-3e26e800-b678-11eb-995e-20530c2c05a6.png)

## Training on improved method
![image](https://user-images.githubusercontent.com/33721483/118407337-42eb9c00-b678-11eb-8011-710b81020391.png)


## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Installation
```
pip install https://github.com/dx199771/Deeper-Networks-for-Image-Classification.git
cd Deeper-Network-for-Image-Classification
python main.py
```

## Details of training
Epoch: 40 (CIFAR-10), 5 (MNIST)

Batch size: 128

Optimizer: Adam optimizer

Learning rate: 0.01 (CIFAR-10), 0.0001 (MNIST)
