General
--------

.. warning::

    Currently, the package handles only NeST VNN inputs. See `NeST VNN <inputs_nestvnn.html>`_

Feature files
~~~~~~~~~~~~~~
The tool accepts one or multiple feature files. As for now, the tool only handles NEST VNN inputs.

Training
~~~~~~~~~

- ``training_data.txt``:
    File with training data.

- ``hierarchy.cx2``:
    Hierarchy in HCX format used to create a visible neural network.


Prediction
~~~~~~~~~~~

- ``test_data.txt``:
    File with testing data.

- ``model_final.pt``:
    The trained model.

Annotation
~~~~~~~~~~~

- 1 or more RO-crates with a file with interpretation scores and with the hierarchy to annotate.
