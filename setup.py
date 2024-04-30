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

requirements = ['cellmaps_utils==0.1.0a23',
                'cellmaps_generate_hierarchy==0.1.0a20',
                'ndex2>=3.8.0,<4.0.0',
                'optuna',
                'scikit-learn',
                'networkx',
                'pandas',
                'torch',
                'torchvision',
                'torchaudio',
                'scipy',
                'joblib'
                ]

setup_requirements = []

test_requirements = []

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
    description="Python Boilerplate contains all the boilerplate you need to create a Python package with command line",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type = 'text/x-rst',
    include_package_data=True,
    keywords='cellmaps_vnn',
    name='cellmaps_vnn',
    packages=find_packages(include=['cellmaps_vnn']),
    package_dir={'cellmaps_vnn': 'cellmaps_vnn'},
    scripts=[ 'cellmaps_vnn/cellmaps_vnncmd.py'],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/idekerlab/cellmaps_vnn',
    version=version,
    zip_safe=False)
