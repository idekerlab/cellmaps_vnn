#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import os
import re
from setuptools import setup, find_packages


with open(os.path.join('cellmaps_vnn', '__init__.py')) as ver_file:
    for line in ver_file:
        if line.startswith('__version__'):
            version=re.sub("'", "", line[line.index("'"):])

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['cellmaps_utils>=0.4.0,<1.0.0',
                'ndex2>=3.8.0,<4.0.0',
                'optuna>=3.6.0,<4.0.0',
                'scikit-learn>=1.3.0,<1.4.0',
                'networkx>=2.8,<2.9',
                'pandas>=2.0.0,<3.0.0',
                'torch>=2.0.0,<3.0.0',
                'torchvision>=0.15.0,<1.0.0',
                'torchaudio>=2.0.0,<3.0.0',
                'scipy>=1.10.0,<2.0.0',
                'joblib>=1.3.2,<2.0.0'
                ]

setup_requirements = []

setup(
    author="Christopher Churas",
    author_email='tools@cm4ai.org',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    description="The Cell Maps VNN Tool enables creation, training, and usage of an interpretable neural "
                "network-based models that predict cell response to a drug.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type = 'text/x-rst',
    include_package_data=True,
    keywords='cellmaps_vnn',
    name='cellmaps_vnn',
    packages=find_packages(include=['cellmaps_vnn']),
    package_dir={'cellmaps_vnn': 'cellmaps_vnn'},
    package_data={'cellmaps_vnn': ['nest_style.cx2', 'interactome_style.cx2']},
    scripts=[ 'cellmaps_vnn/cellmaps_vnncmd.py'],
    setup_requires=setup_requirements,
    url='https://github.com/idekerlab/cellmaps_vnn',
    version=version,
    zip_safe=False)
