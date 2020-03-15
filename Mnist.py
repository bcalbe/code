import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import  Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models



def load_dataset(path = './data'):
    transform = transforms.ToTensor()

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                            shuffle=False, num_workers=2)

    return trainloader, testloader


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out_feature_L1 = self.layer1(x)
        out1 = self.pool1(out_feature_L1)
        out_feature_L2 = self.layer2(out1)
        out = self.pool2(out_feature_L2)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out , out_feature_L1 , out_feature_L2


def train(trainloader,net,save = False): 
    criterion = nn.CrossEntropyLoss()
    # SGD with momentum
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    train_epoch(trainloader,net,save = False)

def train_epoch(trainloader,net,save = False):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # warp them in Variable
            #inputs, labels = Variable(inputs), Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs,_,_ = net(inputs)
            # loss
            loss = criterion(outputs, labels)
            # backward
            loss.backward()
            # update weights
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:  # print every 50 mini-batches
                print('[%d, %5d] average loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
    print("Finished Training")
    if save == True:
        torch.save(net,"Mnist.pkl")


def test(testloader,net):
     #eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    net.eval() 
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs,_,_ = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

def Get_4_and_5():
    sorted_train = {}
    sorted_test = {}
    transform = transforms.ToTensor()

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                            shuffle=False, num_workers=2)

    for image, label in trainloader:
        # label_str = str(label.numpy().item())
        # cur_image = sorted_train.setdefault(label_str,torch.empty(1,1,28,28))
        # sorted_train[label_str] = torch.cat((cur_image,image),0)
        sorted_train.setdefault(str(label.numpy().item()),[]).append(image)
    
    for image, label in testloader:
        sorted_test.setdefault(str(label.numpy().item()),[]).append(image)

    return sorted_train,sorted_test

def train_on_4(sorted_train,net):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    for image in sorted_train['4']:
        optimizer.zero_grad()
        labels = torch.Tensor([4])
        # forward
        outputs,_,_ = net(inputs)
        # loss
        loss = criterion(outputs, labels)
        # backward
        loss.backward()
        # update weights
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:  # print every 50 mini-batches
            print('[%d, %5d] average loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 999))
            running_loss = 0.0
    print("4 finished")

if __name__ == '__main__':
    num_epochs = 1
    num_classes = 10
    batch_size = 100
    learning_rate = 0.001
    train = False
 
    train_loader, test_loader = load_dataset()
    if train == True:
        net = ConvNet()
        train(train_loader,net,True)
    else:
        net = torch.load("Mnist.pkl")

    sorted_train,sorted_test = Get_4_and_5()
    train_on_4(sorted_train,net)
    test(test_loader,net)

    