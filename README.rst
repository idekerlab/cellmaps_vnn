============
cellmaps_vnn
============


.. image:: https://img.shields.io/pypi/v/cellmaps_vnn.svg
        :target: https://pypi.python.org/pypi/cellmaps_vnn

.. image:: https://app.travis-ci.com/idekerlab/cellmaps_vnn.svg
        :target: https://app.travis-ci.com/idekerlab/cellmaps_vnn

.. image:: https://readthedocs.org/projects/cellmaps-vnn/badge/?version=latest
        :target: https://cellmaps-vnn.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status



Cell Maps Visual Neural Network Toolkit

* Free software: MIT license
* Documentation: https://cellmaps-vnn.readthedocs.io.


Dependencies
------------

* `cellmaps_utils <https://pypi.org/project/cellmaps-utils>`__
* `cellmaps_generate_hierarchy <https://pypi.org/project/cellmaps-generate-hierarchy>`__
* `ndex2 <https://pypi.org/project/ndex2>`__
* `optuna <https://pypi.org/project/optuna>`__
* `scikit-learn <https://pypi.org/project/scikit-learn>`__
* `networkx <https://pypi.org/project/networkx>`__
* `pandas <https://pypi.org/project/pandas>`__
* `torch <https://pypi.org/project/torch>`__
* `torchvision <https://pypi.org/project/torchvision>`__
* `torchaudio <https://pypi.org/project/torchaudio>`__

Compatibility
-------------

* Python 3.8+

Installation
------------

.. code-block::

   git clone https://github.com/idekerlab/cellmaps_vnn
   cd cellmaps_vnn
   pip install -r requirements_dev.txt
   make dist
   pip install dist/cellmaps_vnn*whl


Run **make** command with no arguments to see other build/deploy options including creation of Docker image

.. code-block::

   make

Output:

.. code-block::

   clean                remove all build, test, coverage and Python artifacts
   clean-build          remove build artifacts
   clean-pyc            remove Python file artifacts
   clean-test           remove test and coverage artifacts
   lint                 check style with flake8
   test                 run tests quickly with the default Python
   test-all             run tests on every Python version with tox
   coverage             check code coverage quickly with the default Python
   docs                 generate Sphinx HTML documentation, including API docs
   servedocs            compile the docs watching for changes
   testrelease          package and upload a TEST release
   release              package and upload a release
   dist                 builds source and wheel package
   install              install the package to the active Python's site-packages
   dockerbuild          build docker image and store in local repository
   dockerpush           push image to dockerhub

Before running tests and builds, please install ``pip install -r requirements_dev.txt``

For developers
-------------------------------------------

To deploy development versions of this package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below are steps to make changes to this code base, deploy, and then run
against those changes.

#. Make changes

   Modify code in this repo as desired

#. Build and deploy

.. code-block::

    # From base directory of this repo cellmaps_vnn
    pip uninstall cellmaps_vnn -y ; make clean dist; pip install dist/cellmaps_vnn*wh


Needed files
------------

See the format of the files and examples in examples directory of this repository

- gene2ind.txt: A tab-delimited file where the 1st column is index of genes and the 2nd column is the name of genes.
- cell2ind.txt: A tab-delimited file where the 1st column is index of cells and the 2nd column is the name of cells (genotypes).
- cell2mutation.txt: A comma-delimited file where each row has 718 binary values indicating each gene is mutated (1) or not (0). The column index of each gene should match with those in gene2ind.txt file. The line number should match with the indices of cells in cell2ind.txt file.
- cell2cndeletion.txt: A comma-delimited file where each row has 718 binary values indicating copy number deletion (1) (0 for no copy number deletion).
- cell2amplification.txt: A comma-delimited file where each row has 718 binary values indicating copy number amplification (1) (0 for no copy number amplification).
- training_data.txt: A tab-delimited file containing all data points that you want to use to train the model. The 1st column is identification of cells (genotypes), the 2nd column is a SMILES string of the drug and the 3rd column is an observed drug response in a floating point number, and the 4th column is source where the data was obtained from.
- hierarchy.cx2: Hierarchy in HCX format used to create a visible neural network.
- test_data.txt: A tab-delimited file containing all data points that you want to estimate drug response for. The 1st column is identification of cells (genotypes), the 2nd column is a SMILES string of the drug and the 3rd column is an observed drug response in a floating point number, and the 4th column is source where the data was obtained from.

Usage
-----

For information invoke :code:`cellmaps_vnncmd.py -h`

The tool can be used in 3 modes: train, predict and annotate.

**Example usage**

.. code-block::

   cellmaps_vnncmd.py train ./outdir_training --inputdir examples --config_file examples/config.yaml

.. code-block::

   cellmaps_vnncmd.py predict ./outdir_prediction --inputdir ./outdir_training --config_file examples/config.yaml

.. code-block::

   cellmaps_vnncmd.py annotate ./outdir_annotation --model_predictions ./outdir_prediction --ndexuser USERNAME --ndexpassword - --parent_network 0b7b8aee-332f-11ef-9621-005056ae23aa --visibility

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _NDEx: http://www.ndexbio.org
