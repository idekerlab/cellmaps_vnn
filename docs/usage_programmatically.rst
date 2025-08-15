In a project
--------------

To use cellmaps_vnn in a project::

    import cellmaps_vnn

Example (for training):

.. code-block::

    from cellmaps_vnn.train import VNNTrain
    from cellmaps_vnn.runner import CellmapsvnnRunner

    train_cmd = VNNTrain(
        outdir="<output_directory>",
        inputdir="<input_directory>",
        training_data="<training_data_file>",
        gene2id="<gene2id_file>",
        cell2id="<cell2id_file>",
        mutations="<mutations_file>",
        cn_deletions="<copy_number_deletions_file>",
        cn_amplifications="<copy_number_amplifications_file>"
    )

    runner = CellmapsvnnRunner(
        outdir="<output_directory>",
        command=train_cmd,
        inputdir="<input_directory>"
    )

    runner.run()

Tutorial
=========

Step by step tutorial on how to use Cell Maps VNN in a project is available in this
`Jupyter Notebook <https://github.com/idekerlab/cellmaps_vnn/blob/main/notebooks/run-vnn-programmatically-in-your-project.ipynb>`__.
