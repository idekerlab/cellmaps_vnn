Cell Maps VNN (Cell Maps Visible Neural Network)
================================================================


.. image:: https://img.shields.io/pypi/v/cellmaps_vnn.svg
        :target: https://pypi.python.org/pypi/cellmaps_vnn

.. image:: https://app.travis-ci.com/idekerlab/cellmaps_vnn.svg
        :target: https://app.travis-ci.com/idekerlab/cellmaps_vnn

.. image:: https://readthedocs.org/projects/cellmaps-vnn/badge/?version=latest
        :target: https://cellmaps-vnn.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

The Cell Maps VNN Tool allows to create an interpretable neural network-based model that predicts cell response to a drug.
The tool uses hierarchy in `HCX <https://cytoscape.org/cx/cx2/hcx-specification/>`__ format generated by
`cellmaps_generate_hierarchy tool <https://cellmaps-generate-hierarchy.readthedocs.io/>`__ to create visible neural network
that incorporates given input features. Then the tools allows training the model and perform predictions of cell response
to a drug.

The tool creates an output directory (for both training and prediction) where results are stored and registered
within `Research Object Crates (RO-Crate) <https://www.researchobject.org/ro-crate>`__ using
the `FAIRSCAPE-cli <https://pypi.org/project/fairscape-cli>`__.

.. warning::

    As for today, the tool allows to erform training and predictions using data
    for `NeST VNN <https://github.com/idekerlab/nest_vnn>`__ project.

**Overview of Cell Maps VNN Flow**

.. image:: images/cellmaps_vnn_general.png
  :alt: Overview of Cell Maps VNN

* Free software: MIT license
* Source code: https://github.com/idekerlab/cellmaps_vnn

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   inputs
   outputs
   modules
   developer
   authors
   history

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
