# Delve: Deep Live Visualization and Evaluation

[![PyPI version](https://badge.fury.io/py/delve.svg)](https://badge.fury.io/py/delve) [![Build Status](https://travis-ci.org/justinshenk/delve.svg?branch=master)](https://travis-ci.org/justinshenk/delve)

Inspect layer saturation for optimizing your PyTorch or Keras models.

Delve is a Python package for visualizing deep learning model training.

Use Delve if you need a lightweight PyTorch extension that:
- Plots live statistics of network activations to TensorBoard
- Performs spectral analysis to identify layer saturation for network pruning
- Is easily extendible and configurable

------------------

## Motivation

Designing a deep neural network involves optimizing over a wide range of parameters and hyperparameters. Delve allows you to visualize your layer saturation during training so you can grow and shrink layers as needed.  

## Demo

![live layer saturation demo](images/layer-saturation-convnet.gif)

![example_fc.gif](images/example_fc.gif)

## Getting Started

```bash
pip install delve
```

### Layer Saturation
Pass a PyTorch model or `Linear` layers to CheckLayerSat:

```python
from delve import CheckLayerSat

model = TwoLayerNet() # PyTorch network
stats = CheckLayerSat('runs', model) #logging directory and input

... # setup data loader

for i, data in enumerate(train_loader):    
    stats.saturation() # output saturation
```

Only fully-connected and convolutional layers are currently supported.

To log the saturation to console, call `stats.saturation()`. For example:

```bash
Regression - SixLayerNet - Hidden layer size 10                        │
loss=0.231825:  68%|████████████████████▎         | 1350/2000 [00:04<00:02, 289.30it/s]│
linear1:  90%|█████████████████████████████████▎   | 90.0/100 [00:00<00:00, 453.47it/s]│
linear2:  18%|██████▊                               | 18.0/100 [00:00<00:00, 90.68it/s]│
linear3:  32%|███████████▊                         | 32.0/100 [00:00<00:00, 161.22it/s]│
linear4:  32%|███████████▊                         | 32.0/100 [00:00<00:00, 161.24it/s]│
linear5:  28%|██████████▎                          | 28.0/100 [00:00<00:00, 141.11it/s]│
linear6:  90%|██████████████████████████████████▏   | 90.0/100 [00:01<00:00, 56.04it/s]
```

#### Optimize neural network topology

Ever wonder how big your fully-connected layers should be? Delve helps you visualize the effect of modifying the layer size on your layer saturation.

For example, see how modifying the hidden layer size of this network affects the second layer saturation but not the first. Multiple runs show that the fully-connected "linear2" layer (light blue is 256-wide and orange is 8-wide) saturation is sensitive to layer size:

![saturation](images/layer1-saturation.png)

![saturation](images/layer2-saturation.png)

### Log spectral analysis

Writes the top 5 eigenvalues of each layer to TensorBoard summaries:

```python
stats = CheckLayerSat('runs', layers, 'spectrum')
```

Other options
![spectrum](images/spectrum.png)

### Intrinsic dimensionality

View the intrinsic dimensionality of models in realtime:


![intrinsic_dimensionality-layer2](images/layer2-intrinsic.png)

This comparison suggests that the 8-unit layer (light blue) is too saturated and that a larger layer is needed.

### Why this name, Delve?

__delve__ (*verb*):

   - reach inside a receptacle and search for something
   - to carry on intensive and thorough research for data, information, or the like
