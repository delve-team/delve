"""
Extract layer saturation
------------------------
Extract layer saturation with Delve.
"""
import torch
from tqdm import trange

from delve import CheckLayerSat


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

for h in [3, 32]:
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, h, 10

    # Create random Tensors to hold inputs and outputs
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)
    x_test = torch.randn(N, D_in)
    y_test = torch.randn(N, D_out)

    # You can watch specific layers by handing them to delve as a list.
    # Also, you can hand over the entire Module-object to delve and let delve search for recordable layers.
    model = TwoLayerNet(D_in, H, D_out)

    x, y, model = x.to(device), y.to(device), model.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    layers = [model.linear1, model.linear2]
    stats = CheckLayerSat('regression/h{}'.format(h),
                          save_to="plotcsv",
                          modules=layers,
                          device=device,
                          stats=["lsat", "lsat_eval"])

    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    steps_iter = trange(2000, desc='steps', leave=True, position=0)
    steps_iter.write("{:^80}".format(
        "Regression - TwoLayerNet - Hidden layer size {}".format(h)))
    for step in steps_iter:
        # training step
        model.train()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # test step
        model.eval()
        y_pred = model(x_test)
        loss_test = loss_fn(y_pred, y_test)

        # update statistics
        steps_iter.set_description('loss=%g' % loss.item())
        stats.add_scalar("train-loss", loss.item())
        stats.add_scalar("test-loss", loss_test.item())

        stats.add_saturations()
    steps_iter.write('\n')
    stats.close()
    steps_iter.close()
