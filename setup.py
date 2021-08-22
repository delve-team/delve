#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

import os
import os.path as path
import re

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    with open(os.path.join(here, *parts), "r", encoding="utf8") as fp:
        return fp.read()

# Get package version
def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['tensorboardX', 'tqdm', "matplotlib", "pandas"]

this_dir = path.abspath(path.dirname(__file__))
with open(os.path.join(this_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='delve',
    version=find_version("delve", "__init__.py"),
    description=
    'Delve lets you monitor PyTorch model layer saturation during training',
    url='https://github.com/delve-team/delve',
    author='Mats L. Richter & Justin Shenk',
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
