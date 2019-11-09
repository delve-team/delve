import torch
from torch.nn import Module


def _change_all_pca_layer_thresholds(network: Module, verbose: bool = False):
    pass


class LinearPCALayer(Module):

    def __init__(self, in_features: int, threshold: float = .99, keepdim: bool = True, verbose: bool = False):
        super(LinearPCALayer, self).__init__()
        self.verbose = verbose
        self.register_buffer('eigenvalues', torch.zeros(in_features))
        self.register_buffer('eigenvectors', torch.zeros((in_features, in_features)))
        self.register_buffer('_threshold', torch.Tensor(threshold))
        self.register_buffer('covariance_matrix', torch.zeros((in_features, in_features)))
        self.register_buffer('running_sum', torch.zeros(in_features))
        self.register_buffer('seen_samples', torch.zeros(1))
        self.transformation_matrix = None
        self.pca_computed = False
        self.keepdim = keepdim

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def set_threshold(self, threshold: float) -> None:
        self._threshold.data = torch.Tensor(threshold)
        self._compute_pca_matrix()

    def _update_covariance(self, x: torch.Tensor) -> None:
        self.covariance_matrix += torch.matmul(x.transpose(0, 1), x)
        self.running_sum += x.sum(dim=0)
        self.seen_samples += x.shape[0]

    def _compute_covariance(self) -> torch.Tensor:
        seen_samples = self.seen_samples
        running_sum = self.running_sum
        cov_mtx = self._cov_mtx
        cov_mtx /= seen_samples - 1
        avg_mtx = torch.ger(running_sum, running_sum)
        avg_mtx /= seen_samples * (seen_samples - 1)
        cov_mtx -= avg_mtx
        return cov_mtx

    def _reset_covariance(self):
        self.covariance_matrix.data = torch.zeros(self.covariance_matrix.shape)
        self.running_sum.data = torch.zeros(self.running_sum.shape)
        self.seen_samples += torch.zeros(self.seen_samples.shape)

    def _compute_eigenspace(self):
        self.eigenvalues.data, self.eigenvectors.data = self._compute_covariance().symeig(True)
        self.eigenvalues.data, idx = self.eigenvalues.sort(descending=True)
        self.eigenvectors.data = self.eigenvectors[:, idx]

    def _compute_pca_matrix(self):
        if self.verbose:
            print('computing covariance for Linear')
        percentages = self.eigenvalues.cumsum(0) / self.eigenvalues.sum()
        eigen_space = self.eigenvectors[:, percentages < self.threshold]
        if self.verbose:
            print(f'Saturation: {round(eigen_space.shape[1] / self.eigenvalues.shape[0], 4)}%', 'Eigenspace has shape', eigen_space.shape)
        self.transformation_matrix: torch.Tensor = eigen_space.matmul(eigen_space.t())
        self.reduced_transformation_matrix: torch.Tensor = eigen_space

    def forward(self, x):
        if self.training:
            self.pca_computed = False
            self._update_covariance(x)
            return x
        else:
            if not self.pca_computed:
                self._compute_pca_matrix()
                self.pca_computed = True
                self._reset_covariance()
            if self.keepdim:
                return x @ self.transformation_matrix.t()
            else:
                return x @ self.reduced_transformation_matrix


class Conv2DPCALayer(LinearPCALayer):

    def __init__(self, in_filters, threshold: float = 0.99, verbose: bool = False):
        super(Conv2DPCALayer, self).__init__(in_features=in_filters, threshold=threshold, keepdim=True, verbose=verbose)
        if verbose:
            print('Added Conv2D PCA Layer')
        self.convolution = torch.nn.Conv2d(in_channels=in_filters,
                                           out_channels=in_filters,
                                           kernel_size=1, stride=1, bias=False)

    def _update_covariance(self, reshaped_batch: torch.Tensor) -> None:
        self._cov.update(reshaped_batch)

    def _compute_pca_matrix(self):
        if self.verbose:
            print('computing covariance for Conv2D')
        super()._compute_covariance()
        weight = torch.nn.Parameter(self.transformation_matrix.unsqueeze(2).unsqueeze(3))
        self.convolution.weight = weight

    def forward(self, x):
        if self.training:
            self.pca_computed = False
            swapped: torch.Tensor = x.permute([1, 0, 2, 3])
            flattened: torch.Tensor = swapped.flatten(1)
            reshaped_batch: torch.Tensor = flattened.permute([1, 0])
            self._update_covariance(reshaped_batch)
            return x
        else:
            if not self.pca_computed:
                self._compute_pca_matrix()
                self._reset_covariance()
                self.pca_computed = True
            return self.convolution(x)
