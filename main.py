"""
    Author: Xu Dong
    Student Number: 200708160
    Email: x.dong@se20.qmul.ac.uk

    School of Electronic Engineering and Computer Science
    Queen Mary University of London, UK
    London, UK
"""
from time import time
import numpy as np
import utils, random, argparse
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.utils.data as Data
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import random_split
from torchvision.utils import make_grid

from nets.densenet import *
from nets.resnet import *
from nets.resnet_improved import *
from nets.vgg import *


parser = argparse.ArgumentParser(description='PyTorch Deeper Networks')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--epoch', default=40, type=int, help='training times')
parser.add_argument('--batch_size', default=128, type=int, help='number of samples in one pass')
parser.add_argument('--log_interval', default=40, type=int, help='log interval between every output and plot')
parser.add_argument('--val_size', default=5000, type=int, help='validation set size')
parser.add_argument('--dataset_name', default="CIFAR10", type=str, help='training dataset name')
parser.add_argument('--download_data', default=True, type=bool, help='whether download training data')

args = parser.parse_args()

def show_sample():
    # display some example dataset images
    for images, _ in train_loader:
        print('images shape:', images.shape)
        plt.figure(figsize=(16,16))
        plt.axis('off')
        plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0))/ 2 + 0.5) #denormalization
        plt.show()
        break


def data_downloader(dataset, download=True):
    # MNIST data transformation without data augmentation
    mnist_data_transform = transforms.Compose([
         transforms.Grayscale(3),
         transforms.Resize([32,32]),
         transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))
                                        ])
    # CIFAR-10 data transformation with data augmentation
    cifar_data_transform_augumentation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # CIFAR-10 data transform without data augmentation
    cifar_data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
                                        ])
    if dataset == "MNIST":
        data = torchvision.datasets.MNIST('./data',
                                             transform=mnist_data_transform,
                                             download=download
                                             )
    if dataset == "MNISTtest":
        data = torchvision.datasets.MNIST('./data',
                                          transform=mnist_data_transform,
                                          download=download,
                                          train=False
                                          )
    if dataset == "CIFAR10":
        data = torchvision.datasets.CIFAR10('./data/CIFAR',
                                     transform=cifar_data_transform_augumentation,
                                     download=download
                                     )
    if dataset == "CIFAR10test":
        data = torchvision.datasets.CIFAR10('./data/CIFAR',
                                     transform=cifar_data_transform,
                                     download=download,
                                     train=False
                                     )
    return data


def validation(validate_loader,validation_loss,validation_error):
    # evaluate on validation set
    correct = 0
    valid_loss = 0
    with torch.no_grad():
        for step,(inputs,labels) in enumerate(validate_loader):
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)

            output = model_loader(inputs)
            loss = criterion(output, labels)
            valid_loss += loss.item()*inputs.size(0)
            # get predicted value
            pred = output.data.max(1, keepdim=True)[1]
            # record all correctly predicted values
            correct += pred.eq(labels.data.view_as(pred)).sum()
    valid_loss = valid_loss / len(validate_loader.sampler) # average loss value

    print('Validation: \t[{}/{} Error: ({:.1f}%)]\tLoss: {:.6f}'.format(
        correct, len(validate_loader.dataset), 100-(100. * correct / len(validate_loader.dataset)),
        valid_loss))
    # recored validation loss and error history
    validation_loss.append(loss)
    validation_error.append(100-(100. * correct / len(validate_loader.dataset)))

    return validation_loss,validation_error


def testing(testing_loader):
    print("Testing.......")
    # record testing history
    total = 0
    correct = 0
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(testing_loader):
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)

            output = model_loader(inputs)
            _, predicted = torch.max(output.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if step == 0: # display some prediction results
                utils.predicted_image(inputs[:25].cpu(), labels[:25], predicted[:25], args.dataset_name)
    print('Testing: \t[{}/{} Error: ({:.1f}%)]'.format(
        correct, len(testing_loader.dataset), 100-(100. * correct / len(testing_loader.dataset))))

    return 100-(100. * correct / len(testing_loader.dataset))


def training(training_loader, train_loss, validation_loss, validation_error):
    # get start time
    start_time = time()
    for step,(inputs, labels) in enumerate(training_loader):
        inputs = Variable(inputs).to(device)
        labels = Variable(labels).to(device)

        optimizer.zero_grad()
        output = model_loader(inputs)

        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()

        if step % args.log_interval == 1:    # print every log_interval mini-batches
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(inputs), len(training_loader.dataset),
                       100. * step / len(training_loader), loss.item()))

            train_loss.append(loss.item())
            validation(validate_loader, validation_loss, validation_error)
    # measure elapsed time
    end_time = time()
    print("Training time on {} batch size: {}s".format(args.batch_size, (end_time - start_time)))
    # save trained model
    torch.save(model_loader.state_dict(), './models/model_{}.pth'.format(model))


# set up some seed
torch.manual_seed(5)
torch.cuda.manual_seed(5)
np.random.seed(5)
random.seed(5)

# if GPU available
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")

# dataset preparation
dataset = data_downloader(args.dataset_name)
test_dataset = data_downloader(args.dataset_name+"test")

# split training set to training and validation subset
train_size = len(dataset) - args.val_size
train_ds, val_ds = random_split(dataset, [train_size, args.val_size])

# dataloader (training, testing, validation)
train_loader = Data.DataLoader(dataset=train_ds, batch_size=args.epoch, shuffle=True)
validate_loader = Data.DataLoader(dataset=val_ds, batch_size=args.epoch, shuffle=False)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=args.epoch*2, shuffle=False)

# define cross entropy loss function
criterion = nn.CrossEntropyLoss()

"""
########################################################
#############HIGHLIGHT FOR IMPROVED METHODS#############
########################################################
    IMPROVED METHOD TO REDUCE OVERFITTING
    USING DATA AUGMENTATION ON CIFAR-10 DATASET
    USING BATCH NORMALIZATION ON VGG MODELS
    USING DROPOUT AND REDUCE LAYERS ON RESNET MODELS
"""
# test on all 7 models
testing_models= ["vgg16_bn","vgg19_bn","improved_resnet18","improved_resnet50","improved_resnet101","densenet121","densenet161"]

"""
    METHODS WITHOUT IMPROVEMENTS
"""
# models without improvements
#testing_models= ["vgg16","vgg19","resnet18","resnet50","resnet101","densenet121","densenet161"]

# learning rate for each models
lr_set = [0.0001, 0.0001, 0.01, 0.01, 0.01, 0.01, 0.01]
# colours for plotting results
colours = ['blue', 'red', 'green', 'cyan', 'plum', 'sienna', 'yellow']
# record all models' error rate
model_error = []

for i in range(len(testing_models)):
    # load model and learning rate for training the model
    lr = lr_set[i]
    model = testing_models[i]
    
    # define loss and error set for recording history
    validation_loss = []
    validation_error = []
    train_loss = []
    
    # training process
    print("Training on model: {}".format(model))
    model_loader = eval(model + "()").to(device)

    # set up optimizer and learning rate scheduler(optional)
    optimizer = torch.optim.AdamW(model_loader.parameters(), lr=lr)
    #optimizer = torch.optim.Adam(model_loader.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(args.epoch):
        # train for one epoch
        training(train_loader, train_loss, validation_loss, validation_error)
        #scheduler.step()
        #print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
    error = testing(test_loader)
    model_error.append(error)

    # plot single model's error and loss
    utils.plot_loss(validation_loss,train_loss, model)
    utils.plot_error(validation_error, model)
    # plot all models' errors and losses in one figure
    utils.plot_all_errors(model,colours[i], validation_error)
    utils.plot_all_losses(model,colours[i], train_loss)

# print all error rates
error_print = "Error rate on vgg16:{}, vgg19:{}, resnet18:{}, resnet50:{}, resnet101:{}, densenet121:{}, densenet161:{}"
print( error_print.format(*model_error))




