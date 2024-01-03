=====
Outputs
=====

The tool creates several files and folders in the specified output directory.
Below is the list and description of each output generated by the tool.

Training
---------

- ``model_final.pt``:
    The trained model.

- ``std.txt``:
    Standard deviation values for a given training data based on the specified z-score method ('zscore' and 'robustz').

Prediction
-----------

- ``predict.txt``:
    File containing prediction results.

- ``predict_feature_grad_0.txt`` (``predict_feature_grad_1.txt``, ``predict_feature_grad_2.txt``):
    Files containing the gradients for each feature.

- ``hidden`` directory:
    Directory with files containing the gradients of the hidden layer outputs.

Logs and Metadata
-----------------

- ``output.log``:
    A standard log file recording events, errors, and other messages during the execution of the tool.

- ``error.log``:
    A specialized log file recording only error messages encountered during the execution of the tool.