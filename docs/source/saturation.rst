.. _Saturation Overview:

Saturation
==========

`Saturation` is a metric used for identifying the intrinsic dimensionality of
features in a layer.

A visualization of how saturation changes over training and can be used to optimize network topology is provided at https://github.com/justinshenk/playground:

.. image:: _static/saturation_demo.gif

Covariance matrix of features is computed online:

.. math::

    Q(Z_l, Z_l) = \frac{\sum^{B}_{b=0}A_{l,b}^T A_{l,b}}{n} -(\bar{A}_l \bigotimes \bar{A}_l)

for :math:`B` batches of layer output matrix :math:`A_l` and :math:`n` number of samples.

.. note::

    For more information about how saturation is computed, read `"Feature Space Saturation during Training" <https://arxiv.org/abs/2006.08679>`_.
