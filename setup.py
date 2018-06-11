#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

try:
    import torch
except:
    raise ImportError(
        "Install Pytorch via conda or pip first:\n   conda install pytorch -c pytorch"
    )

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'tensorboardX',
]

setup(
    name='delve',
    version='0.1.0',
    description='Delve lets you view Pytorch layer saturation statistics',
    url='https://github.com/justinshenk/delve',
    author='Justin Shenk',
    author_email='shenk.justin@gmail.com',
    long_description=open("README.md").read(),
    license='MIT license',
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False)
