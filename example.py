import delve
import logging
import torch
import torch.nn as nn

from delve import CheckLayerSat
from torch.autograd import Variable # torch0.3

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 30, 100

# Create random Tensors to hold inputs and outputs
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out))

torch.manual_seed(1)

for h in [5,8,100]:
    N, D_in, H, D_out = 64, 1000, h, 10

    # Create random Tensors to hold inputs and outputs
    x = Variable(torch.randn(N, D_in))
    y = Variable(torch.randn(N, D_out))

    model = TwoLayerNet(D_in, H, D_out)
    layers = [model.linear1, model.linear2]
    stats = CheckLayerSat('regression/h{}'.format(h), layers)

    loss_fn = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for t in range(2000):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = loss_fn(y_pred, y)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        stats.writer.add_scalar('loss', loss.data, t)

        optimizer.step()
    stats.writer.close()
