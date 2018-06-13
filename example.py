import delve
import logging
import torch
import torch.nn as nn

from delve import CheckLayerSat
from torch.autograd import Variable
from tqdm import tqdm, trange

import time

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


cuda = torch.cuda.is_available()
torch.manual_seed(1)

for h in [10, 100, 300]:
    start = time.time()

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, h, 10

    # Create random Tensors to hold inputs and outputs
    x = Variable(torch.randn(N, D_in))
    y = Variable(torch.randn(N, D_out))

    model = TwoLayerNet(D_in, H, D_out)

    if cuda:
        x, y, model = x.cuda(), y.cuda(), model.cuda()

    layers = [model.linear1, model.linear2]
    stats = CheckLayerSat('regression/h{}'.format(h), layers)

    loss_fn = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    steps_iter = trange(2000, desc='steps', leave=True, position=0)
    steps_iter.write("{:^80}".format("Regression - TwoLayerNet - Hidden layer size {}".format(h)))
    for i in steps_iter:
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        steps_iter.set_description('loss=%g' % loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stats.saturation()
    end = time.time()
    print("{:.2f} seconds".format(end-start))
    steps_iter.write('\n')
    stats.close()
    steps_iter.close()

