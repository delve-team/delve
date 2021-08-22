---
title: 'Delve: Neural Network Architecture Inspection'
tags:
  - Python
  - deep learning
  - saturation
authors:
  - name: Justin Shenk
    orcid: 0000-0002-0664-7337
    affiliation: "1,2"
  - name: Mats L. Richter
    affiliation: 3
    orcid: 0000-0002-9525-9730
  - name: Wolf Byttner
    affiliation: 4
    orcid: 0000-0002-9525-9730
affiliations:
 - name: VisioLab, Berlin, Germany
   index: 1
 - name: Peltarion, Stockholm, Sweden
   index: 2
 - name: Institute of Cognitive Science, University of Osnabrueck, Osnabrueck, Germany
   index: 3
 - name: Rapid Health, London, England, United Kingdom
   index: 4
date: 16 August 2021
bibliography: paper.bib
---

# Summary
Designing neural networks is a complex task.

Several tools exist which allow analayzing neural networks after and during training.
Tools such as ... allow ...

[Limitation of these methods]

Delve is a Python package for statistical analysis of neural network layer eigenspaces.
Delve hooks into PyTorch [@pytorch] models and allows saving statistics via TensorBoard [@tensorflow2015-whitepaper] events or CSV writers. 
A comprehensive source of documentation is provided on the home page
([http://delve-docs.readthedocs.io](delve-docs.readthedocs.io)).

## Statement of Need
Research on changes in neural network representations has exploded in the past years. [add citations]
Furthermore, researchers who are interested in developing novel algorithms must implement from scratch much of the computational and algorithmic infrastructure for analysis and visualization.
By packaging a library that is particularly useful for extracting statistics from neuarl network training, future researchers can benefit from access to a high-level interface and clearly documented methods for their work.

## Overview of the Library
The software is structured into several modules which distribute tasks. Full details are available at <https://delve-docs.readthedocs.io/>. The ... module provides ...

Subclassing the TensorBoardX `SummaryWriter` [@tensorflow2015-whitepaper]...

## Eigendecomposition of the feature covariance matrix

Covariance matrix of features is computed as described in [@feature-space]...

$$Q(Z_l, Z_l) = \frac{\sum^{B}_{b=0}A_{l,b}^T A_{l,b}}{n} -(\bar{A}_l \bigotimes \bar{A}_l)$$

for $B$ batches of layer output matrix $A_l$ and $n$ number of samples.

# References
