#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

import os
import os.path as path

# Get package version
exec(open('delve/version.py', 'r').read())

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['tensorboardX', 'tqdm', 'MDP']

this_dir = path.abspath(path.dirname(__file__))
with open(os.path.join(this_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='delve',
    version=__version__,
    description=
    'Delve lets you monitor PyTorch model layer saturation during training',
    url='https://github.com/delve-team/delve',
    author='Justin Shenk',
    author_email='shenk.justin@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT license',
    install_requires=requirements,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    python_requires='!= 3.0.*, != 3.1.*',
    packages=find_packages(),
    include_package_data=True,
    keywords='deep learning layer saturation pruning spectral tensorboard network',
    zip_safe=False,
)
