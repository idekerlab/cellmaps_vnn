=====
Usage
=====

The tool cellmaps_vnn allows to build Visible Neural Network (VNN) using hierarchy in CX2 format,
train the network with provided training data and perform predictions.

For training, it takes as an input a directory containing hierarchy and parent network
(the output of cellmaps_generate_hierarchy tool) and text file with training data. For prediction,
it take as an input a directory containing trained model (the output of train mode) and text file with prediction data.

In a project
--------------

To use cellmaps_vnn in a project::

    import cellmaps_vnn

On the command line
---------------------

For information invoke :code:`cellmaps_vnncmd.py -h`

**Usage**

Training:

.. code-block::

  cellmaps_vnncmd.py [--provenance PROVENANCE_PATH] train OUTPUT_DIRECTORY --inputdir HIERARCHY_DIR
        --training_data TRAINING_DATA --gene2id GENE2ID_FILE --cell2id CELL2ID_FILE --mutations MUTATIONS_FILE
        --cn_deletions CN_DELETIONS_FILE --cn_amplifications CN_AMPLIFICATIONS_FILE [OPTIONS]

Prediction:

.. code-block::

  cellmaps_vnncmd.py [--provenance PROVENANCE_PATH] predict OUTPUT_DIRECTORY --inputdir MODEL_DIR
        --predict_data PREDICTION_DATA --gene2id GENE2ID_FILE --cell2id CELL2ID_FILE --mutations MUTATIONS_FILE
        --cn_deletions CN_DELETIONS_FILE --cn_amplifications CN_AMPLIFICATIONS_FILE [OPTIONS]

**Arguments**

For both training and prediction:

TODO

Training:

TODO

Prediction:

TODO

Via Docker
---------------

**Example usage**

.. code-block::

   Coming soon ...


