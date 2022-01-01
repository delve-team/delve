---
title: 'Delve: Neural Network Feature Variance Analysis'
tags:
  - Python
  - deep learning
  - machine learning
  - saturation
  - pytorch
  - AI
authors:
  - name: Justin Shenk^[co-first author]
    orcid: 0000-0002-0664-7337
    affiliation: "1,2"
  - name: Mats L. Richter^[co-first author]
    affiliation: 2
    orcid: 0000-0002-9525-9730
  - name: Wolf Byttner
    affiliation: 3
    orcid: 0000-0002-9525-9730
affiliations:
 - name: VisioLab, Berlin, Germany
   index: 1
 - name: Institute of Cognitive Science, University of Osnabrueck, Osnabrueck, Germany
   index: 2
 - name: Rapid Health, London, England, United Kingdom
   index: 3
date: 16 August 2021
bibliography: paper.bib
---

# Summary
Designing neural networks is a complex task.
Deep neural networks are often referred to as "black box" models - little insight in the function they approximate is gained from looking at the structure of layer outputs.
``Delve`` is a tool for looking at how a neural network represents data, and how these representations, or features, change throughout training.
This tool enables deep learning researchers to understand the limitations and suggest improvements for the design of their networks, such as removing or adding layers.

Several tools exist which allow analyzing neural networks after and during training.
These techniques can be characterized by their focus on either data or model as well as their level of abstractness.
Examples for abstract model-oriented techniques are tools for analyzing the sharpness of local optima [@keskar;@sensitivitygoogle], which can be an indicator for the generalizing capabilities of the trained models.
In these scenarios the complexity of the dataset and model is reduced to the error surface, allowing for insights into the differences between different setups.
A less abstract data-centric technique GradCam by Selvaraju et al. [@gradcam;@gradcamplusplus], reduces the model to a set of class-activation maps that can be overlayed over individual data points to get an intuitive understanding of the inference process.
SVCCA [@svcca;@svcca2] can be considered model-centric and a middle ground in terms of abstractness, since it allows the comparative analysis of the features extracted by specific layers.
SVCCA is also relevant from a functional perspective for this work, since it uses singular value decomposition as a core technique to obtain the analysis results.
Another model-centric tool that allows for a layer-by-layer analysis is logistic regression probes [@alain2016], which utilize logistic regressions trained on the output of a hidden layer to measure the linear separability of the data and thus the quality of the intermediate solution quality of a classifier model.

The latter is of great importance for this work since logistic regression probes are often used to compare models and identify the contribution of layers to overall performance [@feature-space;@sizematters;@goingdeeper] and to demonstrate that the saturation metric is capable of showing parameter-inefficiencies in neural network architectures.

However, the aforementioned tools have significant limitations in terms of their usefulness in practical application scenarios, where these tools are to be used to improve the performance of a given model.
In the case of data-centric tools like GradCam, the solution propagates back to the data, which makes it hard to derive decisions regarding the neural architecture.
However, the biggest concern in all aforementioned tools is the cost of computational resources and the integration of the analysis into the workflow of a deep learning practitioner.
Tools like SVCCA and logistic regression probes require complex and computationally expensive procedures that need to be conducted after training.
This naturally limits these techniques to small benchmarks and primarily academic datasets like Cifar10 [@feature-space].
An analysis tool that is to be used during the development of a deep learning-based model needs to be able to be used with little computational and workflow overhead as possible.
Ideally, the analysis can be done live while the training is in progress, allowing the researcher to interrupt potentially long-running training sessions to improve the model.
Saturation was proposed in 2018 [@Shenk:Thesis:2018] and later refined [@feature-space] and is the only known analysis technique known to the authors that has this capability while allowing to identify parameter-inefficiencies in the setup [@feature-space;@sizematters;@goingdeeper].
To make saturation usable in an application scenario, it is necessary to provide an easy-to-use framework that allows for an integration of the tool into the normal training and inference code with only minimally invasive changes.
It is also necessary that the computation and analysis can be done online as part of the regular forward pass of the model, to make the integration as seamless as possible.
A numerical comparison of these various methods is a promising avenue for future research into model introspection.

``Delve`` provides a framework for allowing a seamless and minimal overhead integration for saturation and
other statistical analysis of neural network layer eigenspaces.
It hooks into PyTorch [@pytorch] models and allows saving statistics via TensorBoard [@tensorflow2015-whitepaper] events or CSV writers.
A comprehensive source of documentation is provided on the home page
([http://delve-docs.readthedocs.io](delve-docs.readthedocs.io)).


## Statement of Need
Research on changes in neural network representations has exploded in the past years [@svcca;@svcca2;@gradcam;@kernelPCA;@alain2016;@featureAttribution].
Furthermore, researchers who are interested in developing novel algorithms must implement from scratch much of the computational and algorithmic infrastructure for analysis and visualization.
By packaging a library that is particularly useful for extracting statistics from neural network training, future researchers can benefit from access to a high-level interface and clearly documented methods for their work.
``Delve`` has already been used in a number of scientific publications [@feature-space;@sizematters;@goingdeeper].
The combination of ease of usage and extensibility in ``Delve`` enables exciting scientific explorations for machine learning researchers and engineers.
The source code for ``Delve`` has been archived to Zenodo with the linked DOI: [@zenodo]

## Overview of the Library
The software is structured into several modules which distribute tasks. Full details are available at <https://delve-docs.readthedocs.io/>.

The TensorBoardX `SummaryWriter` [@tensorflow2015-whitepaper] is used to efficiently save artifacts like images or statistics during training with minimal interruption.
A variety of layer feature statistics can be observed:

| Statistic |
|----------------------------------------------------------------------------------|
| intrinsic dimensionality                                                               |
| layer saturation (intrinsic dimensionality divided by feature space dimensionality      |
| the covariance-matrix                    |
| the determinant of the covariance matrix (also known as generalized variance)          |
| the trace of the covariance matrix, a measure of the variance of the data                  |
| the trace of the diagonal matrix, another way of measuring the dispersion of the data. |
| layer saturation (intrinsic dimensionality divided by feature space dimensionality)    |


Several layers are currently supported:

* Convolutional
* Linear
* LSTM

Additional layers such as PyTorch's ConvTranspose2D are planned for future development (see issue [#43](https://github.com/delve-team/delve/issues/43)).

## Eigendecomposition of the feature covariance matrix
The computation of saturation and other related metrics like the intrinsic dimensionality require the covariance matrix
of the layers output.
Computing the covariance matrix of a layers output on the training or evaluation set is impractical to do naivly, since
it would require to hold the entire dataset in memory.
This would also contradict our goal of seamless integration in existing training loops, which commonly operate with
mini-batches, which are substantially smaller than the entire dataset.
Therefore, an batch-wise approximation algorithm is used in order to compute the covariance matrix life during training:


We compute the covariance matrix $Q(Z_l,Z_l)$, where $Z_l := \sum^{n}_{i=1}(z_{l,i})/n$, using the covariance approximation algorithm between two random variables $X$ and $Y$ with $n$ samples:
$$Q(X, Y) = \frac{\sum^{n}_{i=1} x_i y_i}{n} - \frac{(\sum^{n}_{i=1} x_i)  (\sum^{n}_{i=1} y_i)}{n^2}$$
We make this computation more efficient by exploiting the shape of the layer output matrix $A_l$:
We compute $\sum^{n}_{i=1} x_i y_i$ for all feature combinations in layer $l$ by calculating the running squares $\sum^{B}_{b=0}A_{l,b}^T A_{l,b}$ of the batch output matrices $A_{l,b}$ where $b \in \{0,...,B-1\}$ for $B$ batches. We replace $\frac{(\sum^{n}_{i=1} x_i)  (\sum^{n}_{i=1} y_i)}{n^2}$ by the outer product $\bar{A}_l \bigotimes \bar{A}_l$ of the sample mean $\bar{A}_l$.
This is the running sum of all outputs $z_{l,k}$, where $k \in \{0,...,n\}$ at training time, divided by the total number of training samples $n$.
The final formula for covariance approximation is then:
$$Q(Z_l, Z_l) = \frac{\sum^{B}_{b=0}A_{l,b}^T A_{l,b}}{n} -(\bar{A}_l \bigotimes \bar{A}_l)$$
Since we only store the sum of squares, the running mean and the number of observed samples, we require constant memory and computation is done batch-wise.
The algorithm requires roughly the same number of computations as the processing of a forward pass of the respective layer does; thus we compute saturation after every epoch. The variables are reset at the beginning of each epoch to minimize the bias induced by weight updates during training.
Our algorithm uses a thread-save common value store on a single compute device or node, which furthermore allows to update the covariance matrix asynchronous when the network is trained in a distributed manner.

Another problem is the dimensionality of the data - especially for convolutional layers, where a simple flattening
of the data vector would result in an extreme high dimensional vector and a very expensive singular value composition 
as a direct consequence.

In convolutional layers, we treat every kernel position as an individual observation. This turns an output-tensor of shape (samples $\times$ height $\times$ width $\times$ filters) into a data matrix of shape (samples $\cdot$ height $\cdot$ width $\times$ filters).}
The advantage of this strategy is that no information is lost, while keeping $Q$ at a manageable size.

This approximation method was described alongside the saturation metric in the works of [@Shenk:Thesis:2018;@spectral-analysis] and further refined by [@feature-space].

# References
