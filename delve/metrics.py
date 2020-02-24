from typing import Dict, Union

import numpy as np
import torch


def compute_intrinsic_dimensionality(cov: torch.Tensor, thresh: float = .99) -> int:
    """
    Compute the intrinsic dimensionality based on the covariance matrix
    :param cov: the covariance matrix as a torch tensor
    :param thresh: the delta value for the explained variance of the covariance matrix
    :return: The intrinsic dimensionality, which is a single integer value greater than 0
    """
    eig_vals, eigen_space = cov.symeig(True)
    eig_vals, idx = eig_vals.sort(descending=True)
    eig_vals[eig_vals < 0] = 0
    percentages = eig_vals.cumsum(0) / eig_vals.sum()
    eigen_space = eigen_space[:, percentages < thresh]
    if eigen_space.shape[1] == 0:
        eigen_space = eigen_space[:, :1]
    elif thresh - (percentages[percentages < thresh][-1]) > 0.02:
        eigen_space = eigen_space[:, :eigen_space.shape[1] + 1]
    return eigen_space.shape[1]


def compute_saturation(cov: torch.Tensor, thresh: float = .99) -> float:
    """
    Computes the saturation
    :param cov: the covariance matrix as a torch tensor
    :param thresh: the delta value for the explained variance of the covariance matrix
    :return: a value between 0 and 1
    """
    intrinsic_dimensionality = compute_intrinsic_dimensionality(cov, thresh)
    feature_space_dimensionality = cov.shape[0]

    return intrinsic_dimensionality / feature_space_dimensionality
