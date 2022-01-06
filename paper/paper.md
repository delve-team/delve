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

``Delve`` is a tool for extracting information based on the covariance matrix of the data like saturation and the intrinsic dimensionality from neural network layers.
To emphasize practical usability, special attention is placed on a low overhead and minimally invasive integration of ``Delve`` into
existing training and inference setups:
``Delve`` is directly hooking into PyTorch [@pytorch] models to extract necessary information with little computational and memory overhead, thanks to an efficient covariance approximation algorithm.
We enable the user to store and analyze the extracted statistics without changing their current experiment workflow, by making ``Delve`` easy to integrate into monitoring systems and making this interface easy to expand.
This allows the user to utilize their preferred way of monitoring experiments, from simple CSV-Files and folder structures to more sophisticated solutions like
TensorBoard [@tensorflow2015-whitepaper].
A comprehensive source of documentation is provided on the homepage
([http://delve-docs.readthedocs.io](delve-docs.readthedocs.io)).


## Statement of Need
Research on spectral properties of neural network representations has exploded in the past years [@svcca;@svcca2;@gradcam;@kernelPCA;@alain2016;@featureAttribution].
Publication like [@svcca] and [@feature-space] demonstrate that useful and interesting information can be extracted from the spectral analysis of these latent representations.
It has also been shown that metrics like saturation [@Shenk:Thesis:2018;@spectral-analysis] can be used to optimize neural network architectures by identifying pathological patterns hinting on inefficiencies of the neural network structure.


The main purpose of ``Delve`` is to provide an easy and flexible access to these types of layer-based statistics.
The combination of ease of usage and extensibility in ``Delve`` enables exciting scientific explorations for machine learning researchers and engineers.
``Delve`` has already been used in a number of scientific publications [@feature-space;@sizematters;@goingdeeper].
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
The computation of saturation and other related metrics like the intrinsic dimensionality requires the covariance matrix of the layer's output.
Computing the covariance matrix of a layer's output on the training or evaluation set is impractical to do naively, since it would require holding the entire dataset in memory.
This would also contradict our goal of seamless integration in existing training loops, which commonly operate with mini-batches.
Therefore, a batch-wise approximation algorithm is used to compute the covariance matrix online during training:

We compute the covariance matrix $Q(Z_l,Z_l)$, where $Z_l$ is the output of a layer $l$ by using the covariance approximation algorithm for two random variables $X$ and $Y$ with $n$ samples:
$$Q(X, Y) = \frac{\sum^{n}_{i=1} x_i y_i}{n} - \frac{(\sum^{n}_{i=1} x_i)  (\sum^{n}_{i=1} y_i)}{n^2}$$
The advantage of this method is that only the number of seen samples, the sum of squares and the sum of the variables need to be stored,
making the memory consumption per layer constant with respect to the size of the dataset.
By exploiting the shape of the layer output matrix $A_l$ of the layer $l$ we can compute the covariance of all variable pairs simultaneously:
We can compute $\sum^{n}_{i=1} x_i y_i$ for all feature combinations in layer $l$ by calculating the running squares $\sum^{B}_{b=0}A_{l,b}^T A_{l,b}$ of the batch output matrices $A_{l,b}$ where $b \in \{0,...,B-1\}$ for $B$ batches. We replace $\frac{(\sum^{n}_{i=1} x_i)  (\sum^{n}_{i=1} y_i)}{n^2}$ by the outer product $\bar{A}_l \bigotimes \bar{A}_l$ of the sample mean $\bar{A}_l$.
This is the running sum of all outputs $z_{l,k}$, where $k \in \{0,...,n\}$ at training time, divided by the total number of training samples $n$.
Our formula for a batch-wise approximated covariance matrix can now be written like this:
$$Q(Z_l, Z_l) = \frac{\sum^{B}_{b=0}A_{l,b}^T A_{l,b}}{n} -(\bar{A}_l \bigotimes \bar{A}_l)$$
The batch-wise updating algorithm allows us to integrate the approximation of the covariance matrix as part of the regular forward pass during training and evaluation.
Our algorithm uses a thread-safe common value store on a single compute device or node, which furthermore allows updating the covariance matrix asynchronous when the network is trained in a distributed manner.
To avoid problems that can be caused by rounding errors and numeric instability, our implementation of the algorithm converts by default all data into 64-bit floating-point values.

Another challenge is the dimensionality of the data in convolutional layers, where a simple flattening of the data vector would result in a very high dimensional vector and a computationally expensive singular value decomposition as a direct consequence. To address this issue, we treat every kernel position as an individual observation. This turns an output-tensor of shape (samples $\times$ height $\times$ width $\times$ filters) into a data matrix of shape (samples $\cdot$ height $\cdot$ width $\times$ filters).}
The advantage of this strategy is that no information is lost, while keeping the dimensionality of $Q$ at a manageable size.
Optionally, to reduce the computations required further, the feature map can be automatically reduced in size using linear interpolation to a constant maximum height and width. Since information is lost during this process,
this is disabled by default.

This approximation method was described alongside the saturation metric in the works of [@Shenk:Thesis:2018;@spectral-analysis] and further refined by [@feature-space].

# References
