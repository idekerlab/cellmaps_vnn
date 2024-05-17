Example Usage for NeST VNN
---------------------------

This tool can be used to run the `NeST VNN <https://github.com/idekerlab/nest_vnn>`__ model. The inputs needed for
training the model and performing predictions are described in the `NeST VNN inputs <inputs_nestvnn.html>`_ section
and are located in `examples <https://github.com/idekerlab/cellmaps_vnn/tree/main/examples>`__ directory
in ``cellmaps_vnn`` repository.

Training
~~~~~~~~~

The flow of training NeST VNN

.. image:: images/nest_vnn.png
  :alt: Overview of Cell Maps VNN training flow for NeST VNN

Example run of NeST VNN training using example data provided
in `examples <https://github.com/idekerlab/cellmaps_vnn/tree/main/examples>`__ directory:

.. code-block::

    cellmaps_vnncmd.py train ./6.cellmaps_vnn --inputdir examples --gene2id examples/gene2ind.txt \
        --cell2id examples/cell2ind.txt --training_data examples/training_data.txt --mutations examples/cell2mutation.txt \
        --cn_deletions examples/cell2cndeletion.txt --cn_amplifications examples/cell2cnamplification.txt \
        --genotype_hiddens 4 --lr 0.0005 --epoch 15 --batchsize 64 --optimize 1 --zscore_method auc

Prediction
~~~~~~~~~~~

The flow of prediction and interpretation process using NeST VNN

.. image:: images/nestvnn_pred_int.png
  :alt: Overview of Cell Maps VNN prediction flow for NeST VNN

Example run of NeST VNN prediction and interpretation:

.. code-block::

    cellmaps_vnncmd.py predict ./7.cellmaps_vnn_prediction --inputdir ./6.cellmaps_vnn --gene2id examples/gene2ind.txt \
        --cell2id examples/cell2ind.txt --predict_data examples/test_data.txt --mutations examples/cell2mutation.txt \
        --cn_deletions examples/cell2cndeletion.txt --cn_amplifications examples/cell2cnamplification.txt --batchsize 64 \
        --zscore_method auc

Annotation
~~~~~~~~~~~

The flow of annotation process from  NeST VNN

.. image:: images/nestvnn_annot.png
  :alt: Overview of Cell Maps VNN annotation flow for NeST VNN

.. code-block::

    cellmaps_vnncmd.py annotate ./8.cellmaps_vnn_annotation --model_predictions ./7.cellmaps_vnn_prediction [./7.cellmaps_vnn_prediction_2]
