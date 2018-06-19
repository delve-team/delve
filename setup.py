#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'tensorboardX',
    'tqdm'
]

setup(
    name='delve',
    version='0.1.4',
    description='Delve lets you monitor PyTorch model layer saturation during training',
    url='https://github.com/justinshenk/delve',
    author='Justin Shenk',
    author_email='shenk.justin@gmail.com',
    long_description=open("README.md").read(),
    license='MIT license',
    install_requires=requirements,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
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
    python_requires='>= 2.7, != 3.0.*, != 3.1.*',
    packages=find_packages(),
    include_package_data=True,
    keywords='deep learning layer saturation pruning spectral tensorboard',
    zip_safe=False)
