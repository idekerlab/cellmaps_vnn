NeST VNN
---------
1. Park, S., Silva, E., Singhal, A. et al. A deep learning model of tumor cell architecture elucidates response and
resistance to CDK4/6 inhibitors. Nat Cancer (2024). https://doi.org/10.1038/s43018-024-00740-1

Cell feature files
~~~~~~~~~~~~~~~~~~~
Genetic alteration data: a panel of 718 clinical genes was assembled from the union of genes assessed by FoundationOne
CDx, Tempus xT, PALOMA-3 trial or Project GENIE, each of which assesses mutations and/or copy number aberrations.
To compile genotypes for all cell lines, non-synonymous coding mutations and copy number alterations were extracted for
the 718 clinical panel genes from the Cancer Cell Line Encyclopedia. :sup:`1`

- ``gene2ind.txt``:
    A tab-delimited file where the 1st column is index of genes and the 2nd column is the name of genes.

    .. code-block::

        0	ABCB1
        1	ABCC3
        2	ABL1


- ``cell2ind.txt``:
    A tab-delimited file where the 1st column is index of cells and the 2nd column is the name of cells
    (genotypes).

    .. code-block::

        0	201T_LUNG
        1	22RV1_PROSTATE
        2	2313287_STOMACH

- ``cell2mutation.txt``:
    A comma-delimited file where each row has 718 binary values indicating each gene is mutated (1) or not (0).
    The column index of each gene should match with those in gene2ind.txt file. The line number should match with
    the indices of cells in cell2ind.txt file.

    .. code-block::

        0,0,1,0,0,0..
        0,0,0,0,1,0..
        0,0,0,0,0,0..

- ``cell2cndeletion.txt``:
    A comma-delimited file where each row has 718 binary values indicating copy number deletion (1) (0 for no
    copy number deletion).

    .. code-block::

        0,0,0,0,0,0..
        0,1,0,0,0,0..
        0,0,0,0,1,0..

-  ``cell2amplification.txt``:
    A comma-delimited file where each row has 718 binary values indicating copy number amplification (1) (0 for no
    copy number amplification).

    .. code-block::

        0,0,0,0,0,0..
        0,0,0,1,0,0..
        0,1,0,0,0,0..

Training
~~~~~~~~~
Drug response data were obtained by harmonizing the Cancer Therapeutics Response Portal (CTRP) and
Genomics of Drug Sensitivity in Cancer (GDSC) databases. :sup:`1`

The data from the two datasets were harmonized as follows. Drug information: each molecule’s published name, synonym
or SMILES (Simplified Molecular Input Line Entry System) string was queried using PubChemPy. The associated InChIKey
was extracted and used to identify duplicate drugs (within or between datasets). Cell viability data: for CTRP,
the vehicle control-normalized average percent viability files were used. :sup:`1`

- ``training_data.txt``:
    A tab-delimited file containing all data points that you want to use to train the model. The 1st column is
    identification of cells (genotypes), the 2nd column is a SMILES string of the drug and the 3rd column is
    an observed drug response in a floating point number, and the 4th column is source where the data was obtained from.

    .. code-block::

        HS633T_SOFT_TISSUE	CC1=C(C(=O)N(C2=NC(=NC=C12)NC3=NC=C(C=C3)N4CCNCC4)C5CCCC5)C(=O)C	0.6695136077442607	GDSC2
        KINGS1_CENTRAL_NERVOUS_SYSTEM	CC1=C(C(=O)N(C2=NC(=NC=C12)NC3=NC=C(C=C3)N4CCNCC4)C5CCCC5)C(=O)C	0.6444092636032414	GDSC1

- ``hierarchy.cx2``:
    Hierarchy in HCX format used to create a visible neural network.


Prediction
~~~~~~~~~~~

- ``test_data.txt``:
    A tab-delimited file containing all data points that you want to estimate drug response for. The 1st column is
    identification of cells (genotypes), the 2nd column is a SMILES string of the drug and the 3rd column is
    an observed drug response in a floating point number, and the 4th column is source where the data was obtained from.

    .. code-block::

        EW24_BONE	CC1=C(C(=O)N(C2=NC(=NC=C12)NC3=NC=C(C=C3)N4CCNCC4)C5CCCC5)C(=O)C	0.98852067122827	GDSC1
        OCILY7_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE	CC1=C(C(=O)N(C2=NC(=NC=C12)NC3=NC=C(C=C3)N4CCNCC4)C5CCCC5)C(=O)C	0.2728634745574858	GDSC1


- ``model_final.pt``:
    The trained model.

Annotation
~~~~~~~~~~~

- ``rlipp.out``:
    File with interpretation scores of the predictions made by VNN model. Disease column is optional.

    .. code-block::

        Term	P_rho	P_pval	C_rho	C_pval	RLIPP	Disease
        NEST	9.99800e-01	0.00000e+00	9.33000e-01	4.10702e-147	1.07150e+00	Leukemia
        NEST:6	7.71750e-01	7.47000e-64	7.58600e-01	1.36101e-61	1.01750e+00	Leukemia
        NEST:58	6.44850e-01	1.44552e-38	6.62900e-01	1.62600e-40	9.73000e-01	Leukemia


- ``hierarchy.cx2``:
    Hierarchy in HCX format that will be annotated with interpretation results that will help determine importance of
    the subsystems in the hierarchical network.
