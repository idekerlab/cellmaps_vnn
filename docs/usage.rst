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

**Training:**

.. code-block::

  cellmaps_vnncmd.py [--provenance PROVENANCE_PATH] train OUTPUT_DIRECTORY --inputdir HIERARCHY_DIR
        --training_data TRAINING_DATA --gene2id GENE2ID_FILE --cell2id CELL2ID_FILE --mutations MUTATIONS_FILE
        --cn_deletions CN_DELETIONS_FILE --cn_amplifications CN_AMPLIFICATIONS_FILE [OPTIONS]

**Prediction:**

.. code-block::

  cellmaps_vnncmd.py [--provenance PROVENANCE_PATH] predict OUTPUT_DIRECTORY --inputdir MODEL_DIR
        --predict_data PREDICTION_DATA --gene2id GENE2ID_FILE --cell2id CELL2ID_FILE --mutations MUTATIONS_FILE
        --cn_deletions CN_DELETIONS_FILE --cn_amplifications CN_AMPLIFICATIONS_FILE [OPTIONS]

Arguments
----------

**For both training and prediction:**

- ``outdir``:
    The directory where the output will be written to.

*Required*

- ``--inputdir [HIERARCHY_DIR] [MODEL_DIR]``:
    For training, a directory containing hierarchy and parent network (the output of cellmaps_generate_hierarchy tool).
    For prediction, a directory containing trained model (the output of training mode of cellmaps_vnn).

- ``--training_data TRAINING_DATA`` or ``--predict_data PREDICTION_DATA``:
    For training, the training data to train the model. For prediction, data for which prediction will be performed.

- ``--gene2id GENE2ID_PATH``:
    Gene to ID mapping file.

- ``--cell2id CELL2ID_PATH``:
    Cell to ID mapping file.

- ``--mutations MUTATIONS_PATH``:
    Mutation information for cell lines file.

- ``--cn_deletions CN_DELETIONS_PATH``:
    Copy number deletions for cell lines file.

- ``--cn_amplifications CN_AMPLIFICATIONS_PATH``:
    Copy number amplifications for cell lines file.

*Optional*

- ``--batchsize BATCHSIZE``:
    Defines the number of samples to be processed at a time. Default value is 64.

- ``--cuda CUDA``:
     Indicates the GPU to be used for processing, if available. Default is set to 0.

- ``--zscore_method ZSCORE_METHOD``:
    Specifies the method used for z-scoring in the analysis. Default method is 'auc'.

- ``--std STD_FILENAME``:
    Standardization File name. Default is 'std.txt'.

*Optional for training*

- ``--epoch EPOCH``:
    Training epochs. Defines the total number of training cycles the model will undergo. Default value is 300.

- ``--lr LR``:
    Learning rate. Sets the step size at each iteration while moving toward a minimum of a loss function.
    Default value is 0.001.

- ``--wd WD``:
    Weight decay. Regularization technique by adding a small penalty to the loss function to prevent overfitting.
    Default value is 0.001.

- ``--alpha ALPHA``:
    Loss parameter alpha. Determines the weight given to one part of the loss function in relation to another.
    Default value is 0.3.

- ``--genotype_hiddens GENOTYPE_HIDDENS``:
    Mapping for the number of neurons in each term in genotype parts. Specifies the size of the hidden layers
    in the genotype part of the model. Default value is 4.

- ``--optimize OPTIMIZE``:
    Hyper-parameter optimization. Indicates whether or not hyper-parameter optimization is enabled.
    A value of 1 enables it, and 0 disables it. Default value is 1.

- ``--patience PATIENCE``:
    Early stopping epoch limit. Sets the number of epochs with no improvement after which training will be stopped
    to prevent overfitting. Default value is 30.

- ``--delta DELTA``:
    Minimum change in loss to be considered an improvement. Determines the threshold for regarding
    a change in the loss as significant. Default value is 0.001.

- ``--min_dropout_layer MIN_DROPOUT_LAYER``:
    Start dropout from this Layer number. Specifies the layer number from which to begin applying dropout.
    Default value is 2.

- ``--dropout_fraction DROPOUT_FRACTION``
    Dropout Fraction. Defines the fraction of neurons to drop during the training process to prevent overfitting.
    Default value is 0.3.


Via Docker
---------------

**Example usage**

.. code-block::

   Coming soon ...


