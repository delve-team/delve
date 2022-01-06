import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from tqdm import tqdm

from delve import SaturationTracker

batch_size, bs = 128, 128  # TODO: Duplicate var names to avoid conflicts: Refactor req

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))])

train_set = torchvision.datasets.MNIST(root='./data',
                                       train=True,
                                       download=True,
                                       transform=transform)
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=2,
                                           drop_last=True)

test_set = torchvision.datasets.MNIST(root='./data',
                                      train=False,
                                      download=True,
                                      transform=transform)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=2,
                                          drop_last=True)

# class Net(nn.Module):
#     def __init__(self, h2):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, h2)
#         self.fc3 = nn.Linear(h2, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def flatten(x):
    return to_var(x.view(x.size(0), -1))


class NET(nn.Module):
    def __init__(self,
                 in_dim=784,
                 hidden_dim=bs,
                 n_layers=1,
                 out_dim=784,
                 z_dim=32):

        super(NET, self).__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.encoder_lstm = nn.LSTM(in_dim,
                                    hidden_dim,
                                    n_layers,
                                    batch_first=True)
        # TODO: Support multiple layers in single instance call of LSTM object
        # self.encoder_lstm2 = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.encoder_output = nn.Linear(hidden_dim, z_dim * 2)
        self.decoder_lstm = nn.LSTM(z_dim,
                                    hidden_dim,
                                    n_layers,
                                    batch_first=True)
        self.decoder_output = nn.Linear(hidden_dim, in_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = to_var(torch.randn(*mu.size()))
        z = mu + std * esp
        return z

    def forward(self, x):

        # Encoder
        enc_hidden = self._init_hidden()
        enc, _ = self.encoder_lstm(x, enc_hidden)
        enc = self.relu(enc)

        # Reparameterize
        z_var = self.encoder_output(enc)
        mu, logvar = torch.chunk(z_var, 2, dim=2)  # [bs,c,z_var]  [128,1,10]
        z = self.reparameterize(mu, logvar)

        # Decoder
        dec_hidden = self._init_hidden()
        dec, _ = self.decoder_lstm(z, dec_hidden)
        dec = self.relu(dec)
        dec = self.sigmoid(self.decoder_output(dec))

        return dec, mu, logvar

    def _init_hidden(self):
        return torch.zeros(self.n_layers, bs,
                           self.hidden_dim).cuda(), torch.zeros(
                               self.n_layers, bs, self.hidden_dim).cuda()


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(1)

    epochs = 10

    net = NET()
    if torch.cuda.is_available():
        net.cuda()

    net.to(device)
    logging_dir = 'net/simpson_h2-{}'.format(2)

    stats = SaturationTracker(savefile=logging_dir,
                              save_to='plot',
                              modules=net,
                              include_conv=False,
                              stats=['lsat'],
                              max_samples=1024,
                              verbose=True,
                              writer_args={
                              'figsize': [30, 30],
                              'fontsize': 32
                          },
                              conv_method='mean',
                              device='cpu')

    #net = nn.DataParallel(net, device_ids=['cuda:0', 'cuda:1'])
    eps = torch.Tensor([1e-10]).cuda()

    def loss_fn(recon_x, x, mu, logvar, eps):
        BCE = F.binary_cross_entropy(recon_x + eps, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        return (BCE + KLD) / x.size(0)

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(epochs):
        running_loss = 0.0
        step = 0
        loader = tqdm(train_loader, leave=True,
                      position=0)  # track step progress and loss - optional
        for i, (inputs, labels) in enumerate(loader):
            step = epoch * len(loader) + i
            inputs = flatten(inputs)  # [bs,inp_dim]
            inputs = inputs.unsqueeze(1)  # [bs,1,inp_dim]
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, mu, logvar = net(inputs)
            print(outputs.shape)
            loss = loss_fn(outputs, inputs, mu, logvar, eps)
            loss.backward()

            optimizer.step()

            running_loss += loss.data
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

            # update the training progress display
            loader.set_description(desc='[%d/%d, %5d] loss: %.3f' %
                                   (epoch + 1, epochs, i + 1, loss.data))
            # display layer saturation levels

        stats.add_scalar('epoch', epoch)  # optional
        stats.add_scalar('loss', running_loss.cpu().numpy())  # optional
        stats.add_saturations()

    loader.write('\n')
    loader.close()
    stats.close()
