import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from hessian import hessian
from delve import CheckLayerSat
from tqdm import tqdm, trange
import time

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 128

train_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True,
)

test_set = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False,
)


def get_second_order_grad(grads, xs):
    start = time.time()
    grads2 = []
    for j, (grad, x) in enumerate(zip(grads, xs)):
        print('2nd order on ', j, 'th layer')
        print(x.size())
        grad = torch.reshape(grad, [-1])
        grads2_tmp = []
        for count, g in enumerate(grad):
            g2 = torch.autograd.grad(g, x, retain_graph=True)[0]
            g2 = torch.reshape(g2, [-1])
            grads2_tmp.append(g2[count].data.cpu().numpy())
        grads2.append(torch.from_numpy(np.reshape(grads2_tmp, x.size())))#.to(DEVICE_IDS[0]))
        print('Time used is ', time.time() - start)
    return grads2

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



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)

epochs = 5

for h2 in [8, 32, 128]:  # compare various hidden layer sizes
    net = Net(h2=h2)  # instantiate network with hidden layer size `h2`
    #print(list(net.parameters()))
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    logging_dir = 'convNet/h2-{}'.format(h2)
    stats = CheckLayerSat(logging_dir, net)
    stats.write("CIFAR10 ConvNet - Changing fc2 - size {}".format(h2))  # optional
    ys = torch.zeros(1, device=torch.device('cuda'))
    for epoch in range(epochs):
        running_loss = 0.0
        step = 0
        loader = tqdm(
            train_loader, leave=True, position=0
        )  # track step progress and loss - optional
        for i, data in enumerate(loader):
            step = epoch * len(loader) + i
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            ys = loss
            loss.backward(retain_graph=True)
            optimizer.step()

            running_loss += loss.data
            #if i % 2000 == 1999:  # print every 2000 mini-batches
            #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            #    running_loss = 0.0
            #    stats.add_scalar('batch_loss', running_loss, step)  # optional

            # update the training progress display
            loader.set_description(
                desc='[%d/%d, %5d] loss: %.3f' % (epoch + 1, epochs, i + 1, loss.data)
            )
            # display layer saturation levels
            stats.saturation()
            optimizer.zero_grad()
            xs = optimizer.param_groups[0]['params']
           # jacobian = torch.autograd.grad(ys, xs, create_graph=True)  # first order gradient
           # hessian = get_second_order_grad(jacobian, xs) # second order gradient
            hess = hessian(loss, net.parameters())
            print(hess)




    loader.write('\n')
    loader.close()
    stats.close()
