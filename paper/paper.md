---
title: 'Delve: Neural Network Eigenspace Computation and Visualization'
tags:
  - Python
  - deep learning
  - saturation
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

Several tools exist which allow analyzing neural networks after and during training.
These techniques can be characterized by their focus on either data or model as well as their general abstractness.
Examples for abstract model oriented techniques are tools for analyzing the sharpness of local
optima [@keskar;@sensitivitygoogle], which can be an indicator for the generalizing capeabilities of the trained models.
In these scenarios the complex of dataset and model is reduced to the error surface, allowing for insights into the differences between different setups.
A less abstract data-centric technique GradCam by Selvaraju et al. [@gradcam;@gradcamplusplus], which reduce the model to a set of class-activation maps that can be overlayed over individual data points to get an intuitive understanding of the inference process.
SVCCA [@svcca;@svcca2] can be considered model centric and a middle ground in terms of abstractness, since it allows the comparative analysis 
on the features extracted by specific layers.
SVCCA is also relevant from a functional perspective for this work, since it uses singular value decomposition as a core technique to obtain the analysis results.
Another model centric tool that allows for a layer by layer analysis are logistic regression probes [@alain2016], which utilize logistic regressions trained on the output
of a hidden layer to measure the linear seperability of the data and thus the quality of the intermediate solution quality of a classifier model.

The latter is of great importance for this work, since Logistic Regression Probes are often used to compare models and identify the contribution of layers to overall performance [@feature-space;@sizematters;@goingdeeper] and to demonstrate that the saturation metric can is capable of showing parameter-inefficienies in neural netwoork architectures.

However, the aforementioned  tools have significant limitation in terms of their usefulness in practical application scenarios, where these tools 
are to be used to improve the performance of a given model.
In case of data centric tools like GradCam the solution propagates back to the data, which makes it hard to derive decisions regarding the neural architecture.
However, the biggest concern in all aforementioned tools are the cost in computational resources and the integration of the analysis into the workflow
of a deep learning practicioner.
Tools like SVCCA and Logistic Regression Probes require complex and computationally expensive procedures that need to be conducted after training.
This naturally limits these techniques to small benchmarks and primarily academic datasets like Cifar10 [@feature-space].
A analysis tool that is to be used during the development of a deep learning based model needs to be able to be used with little computational and workflow overhead as possible.
Ideally the analysis can be done life while the training is in progress, allowing the researcher to interupt potentially long running training session to improve the model.
Saturation was proposed in 2018 [@Shenk:Thesis:2018] and later refined [@feature-space] and is the only known analysis technique known to the authors 
that has this capability while also allowing to identify parameter-inefficiencies in the setup [@feature-space;@sizematters;@goingdeeper].
In order to make saturation usable an application scenario, it is necessary to provide a easy-to-use framework that allows for an integration of the tool into the normal training and inference code with only minimaly invasive changes.
It is also necessary that the computation and analysis can be done online as part of the regular forward pass of the model, to make the integration as seemless as possible.

The Python package Delve provides a framework for allowing a seemless and minimal overhead integration for saturation and 
other statistical analysis of neural network layer eigenspaces.
Delve hooks into PyTorch [@pytorch] models and allows saving statistics via TensorBoard [@tensorflow2015-whitepaper] events or CSV writers. 
A comprehensive source of documentation is provided on the home page
([http://delve-docs.readthedocs.io](delve-docs.readthedocs.io)).

## Statement of Need
Research on changes in neural network representations has exploded in the past years [@svcca;@svcca2;@gradcam;@kernelPCA;@alain2016;@featureAttribution].
Furthermore, researchers who are interested in developing novel algorithms must implement from scratch much of the computational and algorithmic infrastructure for analysis and visualization.
By packaging a library that is particularly useful for extracting statistics from neuarl network training, future researchers can benefit from access to a high-level interface and clearly documented methods for their work.

## Overview of the Library
The software is structured into several modules which distribute tasks. Full details are available at <https://delve-docs.readthedocs.io/>. The ... module provides ...

Subclassing the TensorBoardX `SummaryWriter` [@tensorflow2015-whitepaper]...

## Eigendecomposition of the feature covariance matrix

Saturation is a measure of the rank of the layer feature eigenspace introduced by [@Shenk:Thesis:2018;@spectral-analysis] and extended for ... [@feature-space].

Covariance matrix of features is computed online as described in [@feature-space]...

$$Q(Z_l, Z_l) = \frac{\sum^{B}_{b=0}A_{l,b}^T A_{l,b}}{n} -(\bar{A}_l \bigotimes \bar{A}_l)$$

for $B$ batches of layer output matrix $A_l$ and $n$ number of samples.

# References
