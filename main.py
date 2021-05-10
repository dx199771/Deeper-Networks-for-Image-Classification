import argparse
import torch
from time import time
import numpy as np
import torchvision
import torch.utils.data as Data
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torchvision
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import math

from nets.vgg import vgg19_bn
from nets.densenet import *
from nets.resnet import *
from nets.vgg import *
import utils
parser = argparse.ArgumentParser(description='PyTorch Deeper Network Examples')

TRAIN_TIMES = 100
BATCH_SIZE = 500
LR = 0.01
log_interval = 10
model = "resnet101"
val_size = 5000

def data_downloader(dataset,download=False):

    mnist_data_transform = transforms.Compose([transforms.Grayscale(3),
                                         transforms.Resize([32,32]),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))
                                        ])
    cifar_data_transform = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize((0.5,), (0.5,))
                                        ])
    if dataset == "MNIST":
        data = torchvision.datasets.MNIST('./data',
                                             transform=mnist_data_transform,
                                             download=download
                                             )
    if dataset == "CIFAR":
        data = torchvision.datasets.CIFAR10('./data/CIFAR',
                                     transform=cifar_data_transform,
                                     download=download
                                     )
    return data




def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# if GPU available
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")

# dataset preparation
dataset = data_downloader("CIFAR")
test_dataset = data_downloader("CIFAR")


train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = Data.DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
validate_loader = Data.DataLoader(dataset=val_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size = BATCH_SIZE)




# load network
model_loader = eval(model+"()").to(device)

optimizer = torch.optim.Adam(model_loader.parameters(), lr = LR)
criterion = nn.CrossEntropyLoss()


for images, _ in train_loader:
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=12).permute((1, 2, 0)))
    plt.show()
    break

"""
# Training Process
# 
#
"""
train_losses = []
train_counter = []

evaluate_losses = []
evaluate_error = []

def predicted_image(input_img,input_label,output_label):
    grid_img = torchvision.utils.make_grid(input_img[:25].cpu(), nrow=5)
    print(grid_img.shape)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()
def evaluator(validate_loader):
    loss = 0
    correct = 0
    with torch.no_grad():
        for step,(inputs,labels) in enumerate(validate_loader):
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)
            output = model_loader(inputs)
            predicted_image(inputs,labels,output)
            loss += criterion(output, labels)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()
            num_batch = step
    loss /= num_batch

    print('Validation: \t[{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
        correct, len(validate_loader.dataset), 100. * correct / len(validate_loader.dataset),
        loss))

    evaluate_losses.append(loss)
    evaluate_error.append(100. * correct / len(validate_loader.dataset))
    utils.plot_loss(evaluate_losses)
    utils.plot_error(evaluate_error)

#def training():

for epoch in range(TRAIN_TIMES):
    start_time = time()
    for step,(inputs, labels) in enumerate(train_loader):
        inputs = Variable(inputs).to(device)
        labels = Variable(labels).to(device)

        optimizer.zero_grad()
        output = model_loader(inputs)

        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()

        if step % log_interval == 1:    # print every log_interval mini-batches
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(inputs), len(train_loader.dataset),
                       100. * step / len(train_loader), loss.item()))

            train_losses.append(loss.item())
            train_counter.append( ##???
                (step * 64) + ((epoch - 1) * len(train_loader.dataset)))
            utils.plot_loss(train_losses)

            # save trained model
            torch.save(model_loader.state_dict(), './models/model_{}.pth'.format(model))
            #torch.save(optimizer.state_dict(), './models/optimizer.pth')

            dataset = './models/model_{}.pth'.format(model)
            #model = eval(model+"()").to(device)
            #model.load_state_dict(torch.load(dataset))
            evaluator(validate_loader)

        end_time = time()
    print("Training time on {} batch size: {}s".format(BATCH_SIZE,(end_time-start_time)))


"""
# Testing Process
#
"""
