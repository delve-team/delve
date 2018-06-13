import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
from tqdm import tqdm, trange

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

bath_size = 128

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

class Net(nn.Module):
    def __init__(self, h2):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, h2)
        self.fc3 = nn.Linear(h2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


torch.manual_seed(1)
cuda = torch.cuda.is_available()

for h2 in [8, 32, 128]:
    net = Net(h2=h2)

    if cuda:
        net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    stats = CheckLayerSat(
        'convNet/h2-{}/subsample_rate1'.format(h2),
        [net.fc1, net.fc2, net.fc3],
        log_interval=100)

    epoch_iter = trange(5, desc='epochs')
    for epoch in epoch_iter:  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        step = 0
        for i, data in enumerate(trainloader, 0):
            step = epoch * len(trainloader) + i
            inputs, labels = data
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1,
                                                running_loss / 2000))
                running_loss = 0.0
            stats.add_scalar('batch_loss', loss.data, step)

        # Validation
        correct = 0
        total = 0
        net.eval()
        val_samples = 300
        for i, data in enumerate(testloader):
            if i * batch_size > val_samples:
                continue
            images, labels = data
            images = Variable(images).cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        stats.add_scalar('val_accuracy', 100 * correct / total, epoch)
        val_accuracy = 100 * corret / total
        epoch_iter.set_description('({0.data} images) val_accuracy={:.2%}'.format(loss, val_accuracy))
