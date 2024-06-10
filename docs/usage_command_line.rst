On the command line
---------------------

For information invoke :code:`cellmaps_vnncmd.py -h`

Training mode, and Prediction and Interpretation mode
======================================================

*Training:*

.. code-block::

  cellmaps_vnncmd.py [--provenance PROVENANCE_PATH] train OUTPUT_DIRECTORY --inputdir HIERARCHY_DIR
        --training_data TRAINING_DATA --gene2id GENE2ID_FILE --cell2id CELL2ID_FILE --mutations MUTATIONS_FILE
        --cn_deletions CN_DELETIONS_FILE --cn_amplifications CN_AMPLIFICATIONS_FILE [OPTIONS]

*Prediction and Interpretation:*

.. code-block::

  cellmaps_vnncmd.py [--provenance PROVENANCE_PATH] predict OUTPUT_DIRECTORY --inputdir MODEL_DIR
        --predict_data PREDICTION_DATA --gene2id GENE2ID_FILE --cell2id CELL2ID_FILE --mutations MUTATIONS_FILE
        --cn_deletions CN_DELETIONS_FILE --cn_amplifications CN_AMPLIFICATIONS_FILE [OPTIONS]

**Arguments**

*Required for both training step, and prediction and interpretation step*

- ``outdir``:
    The directory where the output will be written to.

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

*Optional for both training step, and prediction and interpretation step*

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

- ``--dropout_fraction DROPOUT_FRACTION``:
    Dropout Fraction. Defines the fraction of neurons to drop during the training process to prevent overfitting.
    Default value is 0.3.

*Optional for prediction and interpretation*

- ``--cpu_count``:
    Interpretation part of this step is performed on CPU and can be performed in parallel if more CPUs are available.
    Default is 1.

- ``--drug_count``:
    Number of top performing drugs. Default is 0. If 0 is set, it is set to number of drugs specified in test data.

- ``--genotype_hiddens``:
    Mapping for the number of neurons in each term in genotype parts. Default is 4.

Annotation mode
================

.. code-block::

  cellmaps_vnncmd.py [--provenance PROVENANCE_PATH] annotate OUTPUT_DIRECTORY
        --model_predictions PREDICTION_DIR [PREDICTION_DIR ..] [OPTIONS]

*Required*

- ``outdir``:
    The directory where the output will be written to.

- ``--model_predictions PREDICTION_DIR [PREDICTION_DIR ..]``:
    Path to one or multiple RO-Crate with the predictions and interpretations obtained from predict step.

*Optional*

- ``--disease DISEASE``:
    Specify the disease or cancer type for which the annotations will be performed. This allows the annotation process
    to tailor the results according to the particular disease or cancer type. If not set, prediction scores for
    all diseases will be aggregated. Examples: Leukemia, Brain Cancer, Lymphoma, Sarcoma, Pancreatic Cancer etc.

- ``--hierarchy HIERARCHY``:
    Path to hierarchy file (optional), if not set will look for ``hierarchy.cx2`` file the first RO-Crate passed
    in --model_predictions argument.

*For upload to NDEx*

- ``--parent_network PARENT_NETWORK``:
    Path to interactome (parent network) of the annotated hierarchy needed if uploading hierarchy in HCX format
    to NDEx. If if not set will look for ``hierarchy_parent.cx2`` file the first RO-Crate passed
    in --model_predictions argument.

- ``--ndexserver NDEXSERVER``:
    Server where the hierarchy can be converted to HCX and saved. Default is ``idekerlab.ndexbio.org``.

- ``--ndexuser NDEXUSER``:
    NDEx user account.

- ``--ndexpassword NDEXPASSWORD``:
    NDEx password. This can either be the password itself or ``-`` to interactively type password.

- ``--visibility``:
    If set, makes Hierarchy and interactome network loaded onto NDEx publicly visible.
