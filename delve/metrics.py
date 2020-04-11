from typing import Dict, Union

import numpy as np
import torch


def compute_intrinsic_dimensionality(cov: torch.Tensor,
                                     thresh: float = .99) -> int:
    """
    Compute the intrinsic dimensionality based on the covariance matrix
    :param cov: the covariance matrix as a torch tensor
    :param thresh: delta value; the explained variance of the covariance matrix
    :return: The intrinsic dimensionality; an integer value greater than zero
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
    :param thresh: delta value; the explained variance of the covariance matrix
    :return: a value between 0 and 1
    """
    intrinsic_dimensionality = compute_intrinsic_dimensionality(cov, thresh)
    feature_space_dimensionality = cov.shape[0]

    return intrinsic_dimensionality / feature_space_dimensionality


def compute_cov_determinant(cov: torch.Tensor) -> float:
    """
    Computes the determinant of the covariance matrix (also known as generalized variance)
    :param cov: the covariannce matrix as torch tensor
    :return: the determinant
    """
    return cov.det().unsqueeze(dim=0).cpu().numpy()[0]


def compute_cov_trace(cov: torch.Tensor) -> float:
    """
    Computes the trace of the covariance matrix (also known as total variance)
    :param cov: the covariannce matrix as torch tensor
    :return: the trace
    """
    return cov.trace().unsqueeze(dim=0).cpu().numpy()[0]


def compute_diag_trace(cov: torch.Tensor) -> float:
    """
    Computes the trace of the covariance matrix diagonal matrix
    :param cov: the covariannce matrix as torch tensor
    :return: the trace
    """
    eig_vals, eigen_space = cov.symeig(True)
    eig_vals[eig_vals < 0] = 0
    return eig_vals.sum().unsqueeze(dim=0).cpu().numpy()[0]

