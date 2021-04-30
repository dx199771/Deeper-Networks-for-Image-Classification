import argparse
import torch
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

from nets.vgg import vgg16
parser = argparse.ArgumentParser(description='PyTorch Deeper Network Examples')

TRAIN_TIMES = 5
BATCH_SIZE = 64
LR = 0.001
log_interval = 10


def data_downloader(dataset,download=True):
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
    if dataset == "MNIST":
        data = torchvision.datasets.MNIST('./data',
                                             transform=data_transform,
                                             download=download
                                             )
    if dataset == "CIFAR":
        data = torchvision.datasets.CIFAR10('./data/CIFAR',
                                     transform=data_transform,
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

val_size = 5000

#train_set, val_set = torch.utils.data.random_split(dataset,[int(len(dataset)*4/5),int(len(dataset)*1/5)])
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = Data.DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size = BATCH_SIZE*2)




torchvision.models.vgg11()

# load network
vgg = vgg16().to(device)
optimizer = torch.optim.Adam(vgg.parameters(), lr = LR)
criterion = nn.CrossEntropyLoss()





# Display some training dataset
for images, _ in train_loader:
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    plt.show()
    break

"""
# Training Process
# 
#
"""
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(TRAIN_TIMES + 1)]


def testing():
    test_loss = 0
    correct = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = Variable(data).to(device)
            target = Variable(target).to(device)
            output = vgg(data)
            test_loss += criterion(output, target)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(TRAIN_TIMES):

    for step,(inputs, labels) in enumerate(train_loader):
        inputs = Variable(inputs).to(device)
        labels = Variable(labels).to(device)
        optimizer.zero_grad()

        output = vgg(inputs)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()

        if step % log_interval == 1:    # print every log_interval mini-batches
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(inputs), len(train_loader.dataset),
                       100. * step / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (step * 64) + ((epoch - 1) * len(train_loader.dataset)))
    torch.save(vgg.state_dict(), './model.pth')
    torch.save(optimizer.state_dict(), './optimizer.pth')

"""
# Testing Process
#
"""