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
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader

def imgshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU())
        self.pool2 = nn.MaxPool2d(kernel_size = 2,stride=2)
        self.fc1 = nn.Linear(32 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x_layer1 = self.layer1(x)
        x1 = self.pool1(x_layer1)
        x_layer2 = self.layer2(x1)
        x2 = self.pool2(x_layer2)
        # fully connect
        x2 = x2.view(-1, 32*8*8)
        x_fc1 = F.relu(self.fc1(x2))
        x_fc2 = F.relu(self.fc2(x_fc1))
        x_fc3 = self.fc3(x_fc2)
        return x_fc3


def train(trainloader,net):

    #net = VGG16
    criterion = nn.CrossEntropyLoss()
    # SGD with momentum
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for iter in range(1):
        for epoch in range(5):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data
                # warp them in Variable
                # inputs, labels = Variable(inputs), Variable(labels)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = net(inputs)
                # loss
                loss = criterion(outputs, labels)
                # backward
                loss.backward()
                # update weights
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] average loss: %.3f' %
                        (iter*5+epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
    print("Finished Training")

def test(testloader,net):
    correct = 0
    total = 0
    for data in testloader:
        images,labels = data
        outputs = net(images)
        _,predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))



if __name__ == '__main__':
    train_loader, test_loader = load_dataset()

    # our model
    net = Net()

    train(train_loader,net)

    test(test_loader,net)
    

