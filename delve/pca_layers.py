import torch
from torch.nn import Module
import numpy as np
from torch.nn.functional import interpolate

global num


def rvs(dim=3):
     random_state = np.random
     H = np.eye(dim)
     D = np.ones((dim,))
     for n in range(1, dim):
         x = random_state.normal(size=(dim-n+1,))
         D[n-1] = np.sign(x[0])
         x[0] -= D[n-1]*np.sqrt((x*x).sum())
         # Householder transformation
         Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
         mat = np.eye(dim)
         mat[n-1:, n-1:] = Hx
         H = np.dot(H, mat)
         # Fix the last sign such that the determinant is 1
     D[-1] = (-1)**(1-(dim % 2))*D.prod()
     # Equivalent to np.dot(np.diag(D), H) but faster, apparently
     H = (D*H.T).T
     return H


def change_all_pca_layer_thresholds_and_inject_random_directions(threshold: float, network: Module, verbose: bool = False, device='cpu', include_names: bool = False):
    in_dims = []
    fs_dims = []
    sat = []
    names = []
    lc = {'lin': 0, 'conv': 0}
    for module in network.modules():
        if isinstance(module, LinearPCALayer):
            module.threshold = threshold
            fake_base = rvs(module.fs_dim)[:, :module.in_dim]
            in_dims.append(module.in_dim)
            fs_dims.append(module.fs_dim)
            sat.append(module.sat)
            fake_projection = fake_base @ fake_base.T
            module.transformation_matrix.data = torch.from_numpy(fake_projection.astype('float32')).to(device)
            names.append(f'Linear-{lc["lin"]}')
            lc["lin"] += 1
            if verbose:
                print(f'Changed threshold for layer {module} to {threshold}')
        elif isinstance(module, Conv2DPCALayer):
            module.threshold = threshold
            in_dims.append(module.in_dim)
            fs_dims.append(module.fs_dim)
            sat.append(module.sat)
            fake_base = rvs(module.fs_dim)[:, :module.in_dim]
            fake_projection = fake_base @ fake_base.T
            module.transformation_matrix.data = torch.from_numpy(fake_projection.astype('float32')).to(device)
            weight = torch.nn.Parameter(module.transformation_matrix.unsqueeze(2).unsqueeze(3))
            module.convolution.weight = weight
            names.append(f'Conv-{lc["conv"]}')
            lc['conv'] += 1
            if verbose:
                print(f'Changed threshold for layer {module} to {threshold}')
    if include_names:
        return sat, in_dims, fs_dims, names
    return sat, in_dims, fs_dims


def change_all_pca_layer_thresholds(threshold: float, network: Module, verbose: bool = False):
    in_dims = []
    fs_dims = []
    sat = []
    names = []
    lc = {'lin': 0, 'conv': 0}
    for module in network.modules():
        if isinstance(module, Conv2DPCALayer) or isinstance(module, LinearPCALayer):
            module.threshold = threshold
            in_dims.append(module.in_dim)
            fs_dims.append(module.fs_dim)
            sat.append(module.sat)
            if isinstance(module, Conv2DPCALayer):
                names.append(f'Conv-{lc["conv"]}')
                lc['conv'] += 1
            else:
                names.append(f"Lin-{lc['lin']}")
                lc["lin"] += 1
            if verbose:
                print(f'Changed threshold for layer {module} to {threshold}')
    return sat, in_dims, fs_dims, names


def change_all_pca_layer_centering(centering: bool, network: Module, verbose: bool = False, downsampling=None):
    in_dims = []
    fs_dims = []
    sat = []
    for module in network.modules():
        if isinstance(module, Conv2DPCALayer) or isinstance(module, LinearPCALayer):
            module.centering = centering
            if isinstance(module, Conv2DPCALayer):
                print('Changed downsampling to ', downsampling)
                module.downsampling = downsampling
            in_dims.append(module.in_dim)
            fs_dims.append(module.fs_dim)
            sat.append(module.sat)
            if verbose:
                print(f'Changed threshold for layer {module} to {centering}')
    return sat, in_dims, fs_dims


class LinearPCALayer(Module):

    num = 0

    def __init__(self, in_features: int,
                 threshold: float = .99,
                 keepdim: bool = True,
                 verbose: bool = False,
                 gradient_epoch_start: int = 20,
                 centering: bool = True):
        super(LinearPCALayer, self).__init__()
        self.register_buffer('eigenvalues', torch.zeros(in_features, dtype=torch.float64))
        self.register_buffer('eigenvectors', torch.zeros((in_features, in_features), dtype=torch.float64))
        self.register_buffer('_threshold', torch.Tensor([threshold]).type(torch.float64))
        self.register_buffer('sum_squares', torch.zeros((in_features, in_features), dtype=torch.float64))
        self.register_buffer('seen_samples', torch.zeros(1, dtype=torch.float64))
        self.register_buffer('running_sum', torch.zeros(in_features, dtype=torch.float64))
        self.register_buffer('mean', torch.zeros(in_features, dtype=torch.float32))
        self.keepdim: bool = keepdim
        self.verbose: bool = verbose
        self.pca_computed: bool = True
        self.gradient_epoch = gradient_epoch_start
        self.epoch = 0
        self.name = f'pca{LinearPCALayer.num}'
        LinearPCALayer.num += 1
        self._centering = centering
        self.data_dtype = None

    def is_floating_point(self):
        return False

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, threshold: float) -> None:
        self._threshold.data = torch.Tensor([threshold]).type(torch.float64).to(self.threshold.device)
        self._compute_pca_matrix()

    @property
    def centering(self):
        return self._centering

    @centering.setter
    def centering(self, centring: bool):
        self._centering = centring
        self._compute_pca_matrix()

    def _update_autorcorrelation(self, x: torch.Tensor) -> None:
        if self.data_dtype is None:
            self.data_dtype = x.dtype
        x = x.type(torch.float64)
       # print(x.dtype)
        self.sum_squares.data += torch.matmul(x.transpose(0, 1), x)
        self.running_sum += x.sum(dim=0)
        self.seen_samples.data += x.shape[0]

    def _compute_autorcorrelation(self) -> torch.Tensor:
        tlen = self.seen_samples
        cov_mtx = self.sum_squares
        cov_mtx = cov_mtx/tlen
        avg = self.running_sum / tlen
        if self.centering:
            avg_mtx = torch.ger(avg, avg)
            cov_mtx = cov_mtx - avg_mtx
        return cov_mtx

    def _compute_eigenspace(self):
        self.eigenvalues.data, self.eigenvectors.data = self._compute_autorcorrelation().symeig(True)#.type(self.data_dtype)
        self.eigenvalues.data, idx = self.eigenvalues.sort(descending=True)
        # correct numerical error, matrix must be positivly semi-definitie
        self.eigenvalues[self.eigenvalues < 0] = 0
        self.eigenvectors.data = self.eigenvectors[:, idx]

    def _reset_autorcorrelation(self):
        self.sum_squares.data = torch.zeros(self.sum_squares.shape, dtype=torch.float64).to(self.sum_squares.device)
        self.seen_samples.data = torch.zeros(self.seen_samples.shape, dtype=torch.float64).to(self.sum_squares.device)
        self.running_sum.data = torch.zeros(self.running_sum.shape, dtype=torch.float64).to(self.sum_squares.device)

    def _compute_pca_matrix(self):
        if self.verbose:
            print('computing autorcorrelation for Linear')
            #print('Mean pre-activation vector:', self.mean)
        percentages = self.eigenvalues.cumsum(0) / self.eigenvalues.sum()
        eigen_space = self.eigenvectors[:, percentages < self.threshold]
        if eigen_space.shape[1] == 0:
            eigen_space = self.eigenvectors[:, :1]
            print(f'Detected singularity defaulting to single dimension {eigen_space.shape}')
        elif self.threshold - (percentages[percentages < self.threshold][-1]) > 0.02:
            print(f'Highest cumvar99 is {percentages[percentages < self.threshold][-1]}, extending eigenspace by one dimension for eigenspace of {eigen_space.shape}')
            eigen_space = self.eigenvectors[:, :eigen_space.shape[1]+1]

        sat = round((eigen_space.shape[1] / self.eigenvalues.shape[0])*100, 4)
        fs_dim = eigen_space.shape[0]
        in_dim = eigen_space.shape[1]
        if self.verbose:
            print(f'Saturation: {round(eigen_space.shape[1] / self.eigenvalues.shape[0], 4)}%', 'Eigenspace has shape', eigen_space.shape)
        self.transformation_matrix: torch.Tensor = eigen_space.matmul(eigen_space.t()).type(torch.float32)
        self.reduced_transformation_matrix: torch.Tensor = eigen_space.type(torch.float32)
        self.sat, self.in_dim, self.fs_dim = sat, in_dim, fs_dim

    def forward(self, x):
        if self.training:
            self.pca_computed = False
            self._update_autorcorrelation(x)
            return x
        else:
            if not self.pca_computed:
                self._compute_autorcorrelation()
                self._compute_eigenspace()
                self._compute_pca_matrix()
                self.pca_computed = True
                self._reset_autorcorrelation()
                self.epoch += 1
            if self.keepdim:
                if not self.centering:
                    return x @ self.transformation_matrix.t()
                else:
                    self.mean = self.mean.to(x.device)
                    self.transformation_matrix = self.transformation_matrix.to(x.device)
                    return ((x-self.mean) @ self.transformation_matrix.t()) + self.mean
            else:
                if not self.centering:
                    return x @ self.reduced_transformation_matrix
                else:
                    return ((x-self.mean) @ self.reduced_transformation_matrix) + self.mean


class Conv2DPCALayer(LinearPCALayer):

    def __init__(self, in_filters, threshold: float = 0.99, verbose: bool = True, gradient_epoch_start: int = 20, centering: bool = False, downsampling: int = None):
        super(Conv2DPCALayer, self).__init__(centering=centering, in_features=in_filters, threshold=threshold, keepdim=True, verbose=verbose, gradient_epoch_start=gradient_epoch_start)
        if verbose:
            print('Added Conv2D PCA Layer')
        self.convolution = torch.nn.Conv2d(in_channels=in_filters,
                                           out_channels=in_filters,
                                           kernel_size=1, stride=1, bias=True)
        self.mean_subtracting_convolution = torch.nn.Conv2d(in_channels=in_filters,
                                                              out_channels=in_filters,
                                                              kernel_size=1, stride=1, bias=True)
        self.mean_subtracting_convolution.weight = torch.nn.Parameter(
            torch.zeros((in_filters, in_filters)).unsqueeze(2).unsqueeze(3)
        )
        self.downsampling = downsampling

    def _compute_pca_matrix(self):
        if self.verbose:
            print('computing autorcorrelation for Conv2D')
        super()._compute_pca_matrix()
        # unsequeeze the matrix into 1x1xDxD in order to make it behave like a 1x1 convolution
        weight = torch.nn.Parameter(
            self.transformation_matrix.unsqueeze(2).unsqueeze(3)
        )
        self.convolution.weight = weight

        self.mean_subtracting_convolution.weight = torch.nn.Parameter(
            torch.zeros_like(self.transformation_matrix).unsqueeze(2).unsqueeze(3)
        )

        if self.centering:
            self.convolution.bias = torch.nn.Parameter(
                self.mean.type(torch.float32)
            )
            self.mean_subtracting_convolution.bias = torch.nn.Parameter(
                -self.mean.type(torch.float32)
            )
        else:
            self.convolution.bias = torch.nn.Parameter(
                torch.zeros_like(self.mean)
            )
            self.mean_subtracting_convolution.bias = torch.nn.Parameter(
                torch.zeros_like(self.mean)
            )

    def forward(self, x):
        if self.training:
            self.pca_computed = False
            if self.downsampling is not None:
                x1 = interpolate(x, size=self.downsampling, mode='nearest')
            else:
                x1 = x
            swapped: torch.Tensor = x1.permute([1, 0, 2, 3])
            flattened: torch.Tensor = swapped.flatten(1)
            reshaped_batch: torch.Tensor = flattened.permute([1, 0])
            self._update_autorcorrelation(reshaped_batch)
            return x
        else:
            if not self.pca_computed:
                self._compute_autorcorrelation()
                self._compute_eigenspace()
                self._compute_pca_matrix()
                self._reset_autorcorrelation()
                self.pca_computed = True
            x1 = self.mean_subtracting_convolution(x)
            x = x + x1
            return self.convolution(x)
