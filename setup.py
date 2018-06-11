#!/usr/bin/env python

from setuptools import setup, find_packages

try:
    import torch
except:
    raise ImportError("Install Pytorch via conda or pip first:\n   conda install pytorch -c pytorch")

setup(
    name='delve',
    version='0.1.0',
    description='View Pytorch layer saturation statistics in TensorBoard',
    url='https://github.com/justinshenk/delve',
    author='Justin Shenk',
    author_email='shenk.justin@gmail.com',
    long_description=open("README.md").read(),
    license='MIT',
    install_requires = [
    "tensorboardX",
    ],
    packages=find_packages())
