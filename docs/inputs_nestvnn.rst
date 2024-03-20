NeST VNN
---------

Cell feature files
~~~~~~~~~~~~~~~~~~~

- ``gene2ind.txt``:
    A tab-delimited file where the 1st column is index of genes and the 2nd column is the name of genes.

- ``cell2ind.txt``:
    A tab-delimited file where the 1st column is index of cells and the 2nd column is the name of cells
    (genotypes).

- ``cell2mutation.txt``:
    A comma-delimited file where each row has 718 binary values indicating each gene is mutated (1) or not (0).
    The column index of each gene should match with those in gene2ind.txt file. The line number should match with
    the indices of cells in cell2ind.txt file.

- ``cell2cndeletion.txt``:
    A comma-delimited file where each row has 718 binary values indicating copy number deletion (1) (0 for no
    copy number deletion).

-  ``cell2amplification.txt``:
    A comma-delimited file where each row has 718 binary values indicating copy number amplification (1) (0 for no
    copy number amplification).

Training
~~~~~~~~~

- ``training_data.txt``:
    A tab-delimited file containing all data points that you want to use to train the model. The 1st column is
    identification of cells (genotypes), the 2nd column is a placeholder for drug id and the 3rd column is an observed
    drug response in a floating point number. [TODO: There are 4 columns]

- ``hierarchy.cx2``:
    Hierarchy in HCX format used to create a visible neural network.


Prediction
~~~~~~~~~~~

- ``test_data.txt``:
    A tab-delimited file containing all data points that you want to estimate drug response for.
    The 1st column is identification of cells (genotypes) and the 2nd column is identification of drugs.
    [TODO: There are 4 columns]

- ``model_final.pt``:
    The trained model.
