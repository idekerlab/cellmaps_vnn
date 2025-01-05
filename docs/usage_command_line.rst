On the command line
---------------------

For information invoke :code:`cellmaps_vnncmd.py -h`

Training
==========

.. code-block::

  cellmaps_vnncmd.py [--provenance PROVENANCE_PATH] train OUTPUT_DIRECTORY --inputdir HIERARCHY_DIR
        --training_data TRAINING_DATA --gene2id GENE2ID_FILE --cell2id CELL2ID_FILE --mutations MUTATIONS_FILE
        --cn_deletions CN_DELETIONS_FILE --cn_amplifications CN_AMPLIFICATIONS_FILE [OPTIONS]

The same command using a config file. If flags and config file are used, the values specified as flags override the values in config file.

.. code-block::

  cellmaps_vnncmd.py train OUTPUT_DIRECTORY --inputdir HIERARCHY_DIR --config_file CONFIG_FILE

**Arguments**

*Required*

- ``outdir``:
    The directory where the output will be written to.

- ``--inputdir [HIERARCHY_DIR] [MODEL_DIR]``:
    For training, a directory containing hierarchy and parent network (the output of cellmaps_generate_hierarchy tool).
    For prediction, a directory containing trained model (the output of training mode of cellmaps_vnn).

Most arguments can be set in configuration file. An example configuration file is provided in the GitHub repo
in ``examples`` directory.

- ``--config_file CONFIG_FILE``:
    Config file that can be used to populate arguments for training. If a given argument is set, it will override the default value. (default: None)

If not set in config file following arguments are **required**

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

*Optional (can be set with flags or in the config file, otherwise defaults will be used)*

- ``--gene_attribute_name GENE_ATTRIBUTE_NAME``:
    Name of the node attribute of the hierarchy with list of genes/ proteins of this node. Default: CD_MemberList. (default: CD_MemberList)

- ``--epoch EPOCH``:
    Training epochs. Defines the total number of training cycles the model will undergo. Default value is 300.

- ``--zscore_method ZSCORE_METHOD``:
    Specifies the method used for z-scoring in the analysis. Default method is 'auc'.

- ``--batchsize BATCHSIZE``:
    Defines the number of samples to be processed at a time. Default value is 64.

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

- ``--optimize OPTIMIZE``:
    Hyper-parameter optimization. Indicates whether or not hyper-parameter optimization is enabled.
    A value of 1 enables optimization using optuna optimizer, and 0 disables it. Default value is 0.
    Different optimizers may be implemented in the future.

- ``--n_trials``:
    Number of trials in hyperparameter optimization

- ``--cuda CUDA``:
     Indicates the GPU to be used for processing, if available. Default is set to 0.

-  ``--skip_parent_copy``:
    If set, hierarchy parent (interactome) will not be copied (default: False)

- ``--slurm``:
    If set, slurm script for training will be generated. (default: False)

- ``--use_gpu``:
    If set, slurm script will be adjusted to run on GPU. (default: False) [Use for slurm only.]

- ``--slurm_partition SLURM_PARTITION``:
    Slurm partition. If use_gpu is set, the default is nrnb-gpu. (default: None)

- ``--slurm_account SLURM_ACCOUNT``:
    Slurm account. If use_gpu is set, the default is nrnb-gpu. (default: None)

Hyperparameter Optimization
=============================

.. code-block::

  cellmaps_vnncmd.py train OUTPUT_DIRECTORY --inputdir HIERARCHY_DIR --config_file CONFIG_FILE --optimize 1 --n_trials 50


To perform hyperparameter optimization `optimize` parameter should be set to 1, and parameters to be optimize should be set as list.

Example:

.. code-block::

    batchsize: [16, 32, 64]          # Batch size
    lr: [0.001, 0.002]               # Learning rate
    wd: 0.001                        # Weight decay

If parameter is set as a single value (float, int etc.), it won't be consider for optimization.


Prediction (with explainability)
==================================

.. code-block::

  cellmaps_vnncmd.py [--provenance PROVENANCE_PATH] predict OUTPUT_DIRECTORY --inputdir MODEL_DIR
        --predict_data PREDICTION_DATA --gene2id GENE2ID_FILE --cell2id CELL2ID_FILE --mutations MUTATIONS_FILE
        --cn_deletions CN_DELETIONS_FILE --cn_amplifications CN_AMPLIFICATIONS_FILE [OPTIONS]

The same command using a config file. If flags and config file are used, the values specified as flags override the values in config file.

.. code-block::

  cellmaps_vnncmd.py predict OUTPUT_DIRECTORY --inputdir MODEL_DIR --config_file CONFIG_FILE

**Arguments**

*Required*

- ``outdir``:
    The directory where the output will be written to.

- ``--inputdir [MODEL_DIR]``:
    A directory containing trained model (the output of training of cellmaps_vnn).

Most arguments can be set in configuration file. An example configuration file is provided in the GitHub repo
in ``examples`` directory.

- ``--config_file CONFIG_FILE``:
    Config file that can be used to populate arguments for training (default: None)

If not set in config file following arguments are **required**

- ``--predict_data PREDICTION_DATA``:
    Test data or data for which prediction will be performed.

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

- ``--cpu_count``:
    Interpretation part of this step is performed on CPU and can be performed in parallel if more CPUs are available.
    Default is 1.

- ``--drug_count``:
    Number of top performing drugs. Default is 0. If 0 is set, it is set to number of drugs specified in test data.

- ``--genotype_hiddens``:
    Mapping for the number of neurons in each term in genotype parts. Default is 4.

- ``std``:
    Path to standardization File (if not set standardization file from RO-Crate will be used).

- ``--cuda CUDA``:
     Indicates the GPU to be used for processing, if available. Default is set to 0.

- ``--slurm``:
    If set, slurm script for training will be generated. (default: False)

- ``--use_gpu``:
    If set, slurm script will be adjusted to run on GPU. (default: False) [Use for slurm only.]

- ``--slurm_partition SLURM_PARTITION``:
    Slurm partition. If use_gpu is set, the default is nrnb-gpu. (default: None)

- ``--slurm_account SLURM_ACCOUNT``:
    Slurm account. If use_gpu is set, the default is nrnb-gpu. (default: None)

Annotation
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

- ``--slurm``:
    If set, slurm script for training will be generated. (default: False)

- ``--slurm_partition SLURM_PARTITION``:
    Slurm partition (default: None)

- ``--slurm_account SLURM_ACCOUNT``:
    Slurm account (default: None)

*For upload to NDEx*

- ``--parent_network PARENT_NETWORK``:
    Path to interactome (parent network) of the annotated hierarchy needed if uploading hierarchy in HCX format
    to NDEx. If if not set will look for ``hierarchy_parent.cx2`` file the first RO-Crate passed
    in --model_predictions argument.

- ``--ndexserver NDEXSERVER``:
    Server where the hierarchy can be converted to HCX and saved. Default is ``ndexbio.org``.

- ``--ndexuser NDEXUSER``:
    NDEx user account.

- ``--ndexpassword NDEXPASSWORD``:
    NDEx password. This can either be the password itself or ``-`` to interactively type password.

- ``--visibility``:
    If set, makes Hierarchy and interactome network loaded onto NDEx publicly visible.
