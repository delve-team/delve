# DELVE: Deep Live Visualization and Evaluation

[![PyPI version](https://badge.fury.io/py/delve.svg)](https://badge.fury.io/py/delve)

Inspect layer saturation and spectral data of your PyTorch models.

Delve is a Python package for visualizing deep learning model training data.

Use Delve if you need a PyTorch extension that:
- Plots live statistics of network activations to TensorBoard and to console
- Performs spectral analysis to identify layer saturation
- Is easily extendible and configurable

------------------

## Motivation

Designing a deep neural network involves optimizing over a wide range of parameters and hyperparameters. Delve allows you to visualize your layer saturation during training so you can grow and shrink layers as needed. Here is an example of the output running `example_deep.py`.

![video of training](images/layer-saturation-convnet.gif)

## Getting Started

```bash
pip install delve
```

NOTE: Currently only tested on a Python console, iPython notebook not yet supported.

### Layer Saturation
Pass a PyTorch model (or layers) to CheckLayerSat:

```python
from delve import CheckLayerSat

model = TwoLayerNet() # PyTorch network
layers = [model.linear1, model.linear2]
stats = CheckLayerSat('runs', layers) #log_dir and input
```

Only fully-connected layers are currently supported.

To log the saturation to console, call `stats.saturation()`. For example:

```bash
INFO:delve:Recording layers {'.linear1': Linear(in_features=1000, out_features=5, bias=True), '.linear2': Linear(in_features=5, out_features=10, bias=True)}
INFO:delve:Adding summaries to directory: regression/h5
INFO:delve:{}
INFO:delve:{'.linear1': 80.0, '.linear2': 50.0}
INFO:delve:{'.linear1': 80.0, '.linear2': 50.0}
INFO:delve:{'.linear1': 80.0, '.linear2': 60.0}
INFO:delve:{'.linear1': 80.0, '.linear2': 60.0}
INFO:delve:{'.linear1': 80.0, '.linear2': 70.0}
INFO:delve:{'.linear1': 80.0, '.linear2': 70.0}
```

#### Optimize neural network topology

Ever wonder how big your layer size should be? Delve helps you visualize the effect of modifying the layer size on your layer saturation.

For example, see how modifying the hidden layer size of this network affects the second layer saturation but not the first. Multiple runs show that the fully-connected "linear2" layer (light blue is 256-wide and orange is 8-wide) saturation is sensitive to layer size:

![saturation](images/layer1-saturation.png)

![saturation](images/layer2-saturation.png)

### Spectral analysis

Plot the top 5 eigenvalues of each layer:

```python
stats = CheckLayerSat('runs', layers, 'spectrum')
```

![spectrum](images/spectrum.png)

### Intrinsic dimensionality

View the intrinsic dimensionality of models in realtime:


![intrinsic_dimensionality-layer2](images/layer2-intrinsic.png)

This comparison suggests that the 8-unit layer (light blue) is too saturated and that a larger layer is needed.

### Why this name, Delve?

__delve__ (*verb*):

   - reach inside a receptacle and search for something
   - to carry on intensive and thorough research for data, information, or the like
