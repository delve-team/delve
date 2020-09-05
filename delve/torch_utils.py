import torch


class TorchCovarianceMatrix(object):

    def __init__(self, bias=False, device='cuda:0', save_data=False):
        self.device = device
        self._input_dim = None  # will be set in _init_internals
        # covariance matrix, updated during the training phase
        self._cov_mtx = None
        # average, updated during the training phase
        self._avg = None
        # number of observation so far during the training phase
        self._tlen = 0

        self.bias = bias

        self.save_data = save_data
        if self.save_data:
            self.saved_samples = None

    def _init_internals(self, x: torch.Tensor):
        """Init the internal structures.

        The reason this is not done in the constructor is that we want to be
        able to derive the input dimension and the dtype directly from the
        data this class receives.
        """
        x = x.type(torch.float64).to(device=self.device)
        # init dtype
        if len(x.shape) >1:
            dim = x.shape[1]
        else:
            dim = x.shape[0]

        self._input_dim = dim
        # init covariance matrix
        self._cov_mtx = torch.zeros((dim, dim)).type(torch.float64).to(self.device)
        # init average
        self._avg = torch.zeros((dim)).type(torch.float64).to(self.device)
        self._tlen = 0

    def update(self, x, vae):
        """Update internal structures given a batch of data
        """
        x = x.type(torch.float64).to(device=self.device)
        if self._cov_mtx is None:
            self._init_internals(x)
        # update the covariance matrix, the average and the number of
        # observations (try to do everything inplace)
        if vae:
            if x.dim() == 3: # For single layer of LSTM; TODO: Support for Multiple layers in single LSTM instance
                x = x.squeeze().t()
            elif x.dim() == 2: # Single layer of FC Linear
                x = x.t()
            else:
                x = x.unsqueeze(0).t()

        self._cov_mtx.data += torch.matmul(x.transpose(0, 1), x)
        self._avg.data += x.sum(dim=0)
        self._tlen += x.shape[0]

        if self.save_data:
            if self.saved_samples is not None:
                self.saved_samples = torch.cat([self.saved_samples, x])
            else:
                self.saved_samples = x
            pass

    def fix(self, center=True):
        """Returns the Covariance matrix"""
        # local variables
        tlen = self._tlen
        cov_mtx = self._cov_mtx
        avg = self._avg / tlen
        cov_mtx = cov_mtx / tlen
        if center:
            avg_mtx = torch.ger(avg, avg)
            cov_mtx -= avg_mtx
        return cov_mtx
