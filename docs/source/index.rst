.. Delve documentation master file, created by
   sphinx-quickstart on Sun Aug 22 13:20:36 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Delve Documentation
===================

Delve is library for analyzing eigenspaces of neural networks.

Delve was developed to help researchers identify information flow through a layer.
Specifically, it computes layer `saturation`, a measure of the covariance of features in a layer: :ref:`Saturation Overview`.

It is useful for optimizing neural network topology, particularly identifying over or under-saturated layers.

If you use Delve in your publication, please cite:

.. code-block:: txt

   @software{delve,
   author       = {Justin Shenk and
                     Mats L. Richter and
                     Wolf Byttner and
                     Michał Marcinkiewicz},
   title        = {delve-team/delve: v0.1.45},
   month        = aug,
   year         = 2021,
   publisher    = {Zenodo},
   version      = {v0.1.45},
   doi          = {10.5281/zenodo.5233860},
   url          = {https://doi.org/10.5281/zenodo.5233860}
   }

Delve allows extracting features from neural network layers and computing the eigenspace of several layers.

Supported Layers
----------------

* Dense/Linear
* LSTM
* Convolutional

Statistics
----------

Layer eigenspace computation allows computing information flow between layers, including:

* feature variance
* feature covariance
* layer feature instrinsic dimensionality

Several statistics are supported:

.. code::

   idim        : intrinsic dimensionality
   lsat        : layer saturation (intrinsic dimensionality divided by feature space dimensionality)
   cov         : the covariance-matrix (only saveable using the 'npy' save strategy)
   det         : the determinant of the covariance matrix (also known as generalized variance)
   trc         : the trace of the covariance matrix, generally a more useful metric than det for determining
               the total variance of the data than the determinant.
               However note that this does not take the correlation between
               features into account. On the other hand, in most cases the determinent will be zero, since
               there will be very strongly correlated features, so trace might be the better option.
   dtrc        : the trace of the diagonalmatrix, another way of measuring the dispersion of the data.
   lsat        : layer saturation (intrinsic dimensionality
               divided by feature space dimensionality)
   cov         : the covariance-matrix (only saveable using
               the 'npy' save strategy)
   embed       : samples embedded in the eigenspace of dimension 2


To support researchers, it allows saving plots at various intervals through the :class:`~delve.torchcallback.SaturationTracker` class.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   Installation <install>
   Saturation <saturation>
   Examples Gallery <gallery/index>

.. toctree::
   :maxdepth: 1
   :caption: Examples

   Academic Gallery <gallery>
   Example Plots <examples>

.. toctree::
   :maxdepth: 1
   :caption: Reference Guide

   Reference to All Attributes and Methods <reference>
   Bugs and Support <support>

.. toctree::
   :maxdepth: 1
   :caption: Developer

   Contributing to Delve <contributing>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
