import torch
from tqdm import trange

from delve.torchcallback import CheckLayerSat


class LayerCake(torch.nn.Module):
    def __init__(self, D_in, H1, H2, H3, H4, H5, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(LayerCake, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, H3)
        self.linear4 = torch.nn.Linear(H3, H4)
        self.linear5 = torch.nn.Linear(H4, H5)
        self.linear6 = torch.nn.Linear(H5, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary (differentiable) operations on Tensors.
        """
        x = self.linear1(x).clamp(min=0)
        x = self.linear2(x).clamp(min=0)
        x = self.linear3(x).clamp(min=0)
        x = self.linear4(x).clamp(min=0)
        x = self.linear5(x).clamp(min=0)
        y_pred = self.linear6(x)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, D_out = 64, 100, 10

H1, H2, H3, H4, H5 = 50, 50, 50, 50, 50

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

for h in [10, 100, 300]:

    # Create random Tensors to hold inputs and outputs
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    model = LayerCake(D_in, h, H2, H3, H4, H5, D_out)

    x, y, model = x.to(device), y.to(device), model.to(device)

    stats = CheckLayerSat(
        'regression/h{}'.format(h),
        'csv',
        model,
        device=device,
        reset_covariance=True,
    )

    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    steps_iter = trange(2000, desc='steps', leave=True, position=0)
    steps_iter.write("{:^80}".format(
        "Regression - SixLayerNet - Hidden layer size {}".format(h)))
    for i in steps_iter:
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        steps_iter.set_description('loss=%g' % loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stats.add_saturations()
        #stats.saturation()
    steps_iter.write('\n')
    stats.close()
    steps_iter.close()
