Training and testing multiple models
--------------------------------------

Jupyter Notebook Example
~~~~~~~~~~~~~~~~~~~~~~~~

The notebook example provided in GitHub repository demonstrates how to use cellmaps_vnn with five distinct training
datasets. Each trained model is used to generate predictions, and the resulting system importance scores are aggregated
and visualized on the hierarchy.

-> `Multiple VNN Models Guide <https://github.com/idekerlab/cellmaps_vnn/blob/main/notebooks/multiple-vnn-models-guide.ipynb>`_

Script Example
~~~~~~~~~~~~~~

**Use case**: You want to train and test n models for p drugs with k configurations each.

**Example**: You have two drugs and want to train and test models with 5 different configurations. You can create a
directory for each drug with multiple configuration files for each. Here we have a data directory that contains the
hierarchy used to build the VNN model, and two directories drug1 and drug2 that contain configuration files.

**Inputs**:

.. code-block::

    data/
        ├── hierarchy.cx2
        └── data_for_drug1/
        └── data_for_drug2/
    └── drug1/
        ├── config_file1.yaml
        ├── config_file2.yaml
        ├── config_file3.yaml
        ├── config_file4.yaml
        └── config_file5.yaml
    └── drug2/
        ├── config_file1.yaml
        ├── config_file2.yaml
        ├── config_file3.yaml
        ├── config_file4.yaml
        └── config_file5.yaml

Below is the Bash script that automates the process of training and testing n models for p drugs with k configurations
each. It organizes outputs in a new directory structure with subdirectories for each drug.

.. code-block::

    #!/bin/bash

    DATA_DIR="data" # hierarchy.cx2 should be placed in this directory
    RESULTS_DIR="results"
    DRUGS=("drug1" "drug2") # List of drugs and directory names inside DATA_DIR
    CONFIG_FILES=("config_file1.yaml" "config_file2.yaml") # Config files

    mkdir -p "$RESULTS_DIR"

    # Loop through drugs
    for drug in "${DRUGS[@]}"; do
        echo "Processing $drug..."
        mkdir -p "$RESULTS_DIR/$drug"

        # Loop through configuration files
        for i in "${!CONFIG_FILES[@]}"; do
            config_file="${CONFIG_FILES[$i]}"
            config_index=$((i + 1))

            # Define output directories
            train_outdir="$RESULTS_DIR/$drug/${drug}_${config_index}_train"
            test_outdir="$RESULTS_DIR/$drug/${drug}_${config_index}_test"

            # Training command
            echo "Training $drug with $config_file..."
            cellmaps_vnncmd.py train "$train_outdir" --inputdir "$DATA_DIR" --config_file "$DATA_DIR/$drug/$config_file" --slurm --use_gpu
        train_job_name="${drug}_${config_index}_train"
            train_job_id=$(sbatch --parsable --job-name="$train_job_name" "$train_outdir/cellmapvnntrainjob.sh")

            # Testing command (dependent on training)
            echo "Testing $drug with $config_file..."
            cellmaps_vnncmd.py predict "$test_outdir" --inputdir "$train_outdir" --config_file "$DATA_DIR/$drug/$config_file" --slurm --use_gpu
        test_job_name="${drug}_${config_index}_test"
            sbatch --dependency=afterok:$train_job_id --job-name="$test_job_name" "$test_outdir/cellmapvnnpredictjob.sh"

            echo "Completed $drug configuration $config_index."
        done
    done

    echo "All training and testing processes initiated!"

**Outputs**:

.. code-block::

    results/
    └── drug1/
        ├── drug1_1_train/
        ├── drug1_1_test/
        ├── drug1_2_train/
        ├── drug1_2_test/
        └── ...
    └── drug2/
        ├── drug2_1_train/
        ├── drug2_1_test/
        ├── drug2_2_train/
        ├── drug2_2_test/
        └── ...
