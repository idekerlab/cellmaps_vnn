#! /usr/bin/env python

import os
import time
import logging
from cellmaps_utils import logutils, constants
from cellmaps_utils.provenance import ProvenanceUtil
from cellmaps_vnn.annotate import VNNAnnotate
import cellmaps_vnn.constants as vnnconstants

from cellmaps_vnn.predict import VNNPredict

from cellmaps_vnn.train import VNNTrain

import cellmaps_vnn
from cellmaps_vnn.exceptions import CellmapsvnnError

logger = logging.getLogger(__name__)


class VnnRunner(object):

    def __init__(self, outdir):
        if outdir is None:
            raise CellmapsvnnError('outdir is None')

        self._outdir = os.path.abspath(outdir)

    def run(self):
        """
        Runs VNN
        :raises NotImplementedError: Always raised cause subclasses need to implement
        """
        raise NotImplementedError('subclasses need to implement')


class SLURMCellmapsvnnRunner(VnnRunner):
    def __init__(self, outdir=None,
                 command=None,
                 inputdir=None,
                 gene_attribute_name=vnnconstants.GENE_SET_COLUMN_NAME,
                 gene2id=None,
                 cell2id=None,
                 mutations=None,
                 cn_deletions=None,
                 cn_amplifications=None,
                 training_data=None,
                 batchsize=vnnconstants.DEFAULT_BATCHSIZE,
                 cuda=vnnconstants.DEFAULT_CUDA,
                 zscore_method=vnnconstants.DEFAULT_ZSCORE_METHOD,
                 epoch=VNNTrain.DEFAULT_EPOCH,
                 lr=VNNTrain.DEFAULT_LR,
                 wd=VNNTrain.DEFAULT_WD,
                 alpha=VNNTrain.DEFAULT_ALPHA,
                 genotype_hiddens=vnnconstants.DEFAULT_GENOTYPE_HIDDENS,
                 optimize=VNNTrain.DEFAULT_OPTIMIZE,
                 n_trials=VNNTrain.DEFAULT_N_TRIALS,
                 patience=VNNTrain.DEFAULT_PATIENCE,
                 delta=VNNTrain.DEFAULT_DELTA,
                 min_dropout_layer=VNNTrain.DEFAULT_MIN_DROPOUT_LAYER,
                 dropout_fraction=VNNTrain.DEFAULT_DROPOUT_FRACTION,
                 skip_parent_copy=False,
                 cpu_count=VNNPredict.DEFAULT_CPU_COUNT,
                 drug_count=VNNPredict.DEFAULT_DRUG_COUNT,
                 predict_data=None,
                 std=None,
                 model_predictions=None,
                 disease=None,
                 hierarchy=None,
                 parent_network=None,
                 ndexserver=None,
                 ndexuser=None,
                 ndexpassword=None,
                 visibility=False,
                 gpu=False,
                 slurm_partition=None,
                 slurm_account=None,
                 input_data_dict=None
                 ):
        super().__init__(outdir)
        self._start_time = int(time.time())
        self._command = command
        self._inputdir = inputdir
        self._gene_attribute_name = gene_attribute_name
        self._gene2id = gene2id
        self._cell2id = cell2id
        self._mutations = mutations
        self._cn_deletions = cn_deletions
        self._cn_amplifications = cn_amplifications
        self._training_data = training_data
        self._batchsize = batchsize
        self._cuda = cuda
        self._zscore_method = zscore_method
        self._epoch = epoch
        self._lr = lr
        self._wd = wd
        self._alpha = alpha
        self._genotype_hiddens = genotype_hiddens
        self._optimize = optimize
        self._n_trials = n_trials
        self._patience = patience
        self._delta = delta
        self._min_dropout_layer = min_dropout_layer
        self._dropout_fraction = dropout_fraction
        self._skip_parent_copy = skip_parent_copy
        self._cpu_count = cpu_count
        self._drug_count = drug_count
        self._predict_data = predict_data
        self._std = std
        self._model_predictions = model_predictions
        self._disease = disease
        self._hierarchy = hierarchy
        self._parent_network = parent_network
        self._ndexserver = ndexserver
        self._ndexuser = ndexuser
        self._ndexpassword = ndexpassword
        self._visibility = visibility
        self._gpu = gpu
        self._slurm_partition = slurm_partition
        self._slurm_account = slurm_account
        self._input_data_dict = input_data_dict

    def _write_slurm_directives(self, out,
                                allocated_time='96:00:00',
                                mem='32G', cpus_per_task='4',
                                job_name='cellmaps_vnn'):
        """
        Writes slurm directives

        :param allocated_time:
        :param mem:
        :param cpus_per_task:
        :param job_name:
        :return:
        """
        out.write('#!/bin/bash\n\n')
        out.write('#SBATCH --job-name=' + str(job_name) + '\n')
        out.write('#SBATCH --chdir=' + self._outdir + '\n')

        out.write('#SBATCH --output=%x.%j.out\n')
        if self._slurm_partition is not None:
            out.write('#SBATCH --partition=' + self._slurm_partition + '\n')
        if self._slurm_account is not None:
            out.write('#SBATCH --account=' + self._slurm_account + '\n')
        if self._gpu:
            out.write('#SBATCH --gres=gpu:1\n')
            out.write('#SBATCH --dependency=singleton\n')
        else:
            out.write('#SBATCH --ntasks=1\n')
            out.write('#SBATCH --cpus-per-task=' + str(cpus_per_task) + '\n')
        out.write('#SBATCH --mem=' + str(mem) + '\n')
        out.write('#SBATCH --time=' + str(allocated_time) + '\n\n')

        out.write('echo $SLURM_JOB_ID\n')
        out.write('echo $HOSTNAME\n')

    def _write_job_for_command(self):
        """
        Runs VNN
        :raises NotImplementedError: Always raised cause
                                     subclasses need to implement
        """

        if isinstance(self._command, VNNTrain):
            if self._inputdir is None:
                raise CellmapsvnnError("Input dir with hierarchy is required for training.")
            filename = 'cellmapvnntrainjob.sh'
            with open(os.path.join(self._outdir, filename), 'w') as f:
                self._write_slurm_directives(out=f, job_name='cellmapvnntrain')
                f.write(
                    'cellmaps_vnncmd.py train ' + os.path.join(self._outdir, 'out_train') +
                    ' --inputdir ' + os.path.abspath(self._inputdir) +
                    ' --gene_attribute_name ' + self._gene_attribute_name)
                if self._gene2id:
                    f.write(' --gene2id ' + os.path.abspath(self._gene2id))
                if self._cell2id:
                    f.write(' --cell2id ' + os.path.abspath(self._cell2id))
                if self._mutations:
                    f.write(' --mutations ' + os.path.abspath(self._mutations))
                if self._cn_deletions:
                    f.write(' --cn_deletions ' + os.path.abspath(self._cn_deletions))
                if self._cn_amplifications:
                    f.write(' --cn_amplifications ' + os.path.abspath(self._cn_amplifications))
                if self._training_data:
                    f.write(' --training_data ' + os.path.abspath(self._training_data))
                f.write(
                    ' --batchsize ' + str(self._batchsize) +
                    ' --cuda ' + str(self._cuda) +
                    ' --zscore_method ' + self._zscore_method +
                    ' --epoch ' + str(self._epoch) +
                    ' --lr ' + str(self._lr) +
                    ' --wd ' + str(self._wd) +
                    ' --alpha ' + str(self._alpha) +
                    ' --genotype_hiddens ' + str(self._genotype_hiddens) +
                    ' --optimize ' + str(self._optimize) +
                    ' --n_trials ' + str(self._n_trials) +
                    ' --patience ' + str(self._patience) +
                    ' --delta ' + str(self._delta) +
                    ' --min_dropout_layer ' + str(self._min_dropout_layer) +
                    ' --dropout_fraction ' + str(self._dropout_fraction)
                )
                if self._skip_parent_copy:
                    f.write(' --skip_parent_copy')
                f.write('\n')
                f.write('exit $?\n')

            os.chmod(os.path.join(self._outdir, filename), 0o755)
            return filename

        elif isinstance(self._command, VNNPredict):
            if self._inputdir is None:
                raise CellmapsvnnError("Input dir with trained model is required for prediction.")
            filename = 'cellmapvnnpredictjob.sh'
            with open(os.path.join(self._outdir, filename), 'w') as f:
                self._write_slurm_directives(out=f, job_name='cellmapvnnpredict')
                f.write(
                    'cellmaps_vnncmd.py predict ' + os.path.join(self._outdir, 'out_predict') +
                    ' --inputdir ' + os.path.abspath(self._inputdir))
                if self._gene2id:
                    f.write(' --gene2id ' + os.path.abspath(self._gene2id))
                if self._cell2id:
                    f.write(' --cell2id ' + os.path.abspath(self._cell2id))
                if self._mutations:
                    f.write(' --mutations ' + os.path.abspath(self._mutations))
                if self._cn_deletions:
                    f.write(' --cn_deletions ' + os.path.abspath(self._cn_deletions))
                if self._cn_amplifications:
                    f.write(' --cn_amplifications ' + os.path.abspath(self._cn_amplifications))
                if self._predict_data:
                    f.write(' --predict_data ' + os.path.abspath(self._predict_data))
                f.write(
                    ' --batchsize ' + str(self._batchsize) +
                    ' --cuda ' + str(self._cuda) +
                    ' --zscore_method ' + self._zscore_method +
                    ' --genotype_hiddens ' + str(self._genotype_hiddens) +
                    ' --cpu_count ' + str(self._cpu_count) +
                    ' --drug_count ' + str(self._drug_count)
                )
                if self._std:
                    f.write(' --std ' + os.path.abspath(self._std))
                f.write('\n')
                f.write('exit $?\n')

            os.chmod(os.path.join(self._outdir, filename), 0o755)
            return filename

        elif isinstance(self._command, VNNAnnotate):
            self._gpu = False
            filename = 'cellmapvnnannotatejob.sh'
            with open(os.path.join(self._outdir, filename), 'w') as f:
                self._write_slurm_directives(out=f, job_name='cellmapvnnannotate')
                f.write(
                    'cellmaps_vnncmd.py annotate ' + os.path.join(self._outdir, 'out_annotate') +
                    ' --model_predictions ' + ' '.join(self._model_predictions))
                if self._disease:
                    f.write(' --disease ' + self._disease)
                if self._hierarchy:
                    f.write(' --hierarchy ' + self._hierarchy)
                if self._parent_network:
                    f.write(' --parent_network ' + self._parent_network)
                if self._ndexserver:
                    f.write(' --ndexserver ' + self._ndexserver)
                if self._ndexuser:
                    f.write(' --ndexuser ' + self._ndexuser)
                if self._ndexpassword:
                    f.write(' --ndexpassword ' + self._ndexpassword)
                if self._visibility:
                    f.write(' --visibility')
                f.write('\n')
                f.write('exit $?\n')

            os.chmod(os.path.join(self._outdir, filename), 0o755)
            return filename

        else:
            raise CellmapsvnnError("Command not recognized")

    def run(self):
        """
        Runs CM4AI Pipeline


        :return:
        """
        logger.debug('In run method')

        if not os.path.isdir(self._outdir):
            os.makedirs(self._outdir, mode=0o755)

        logutils.write_task_start_json(outdir=self._outdir,
                                       start_time=self._start_time,
                                       data={
                                           'commandlineargs': self._input_data_dict},
                                       version=cellmaps_vnn.__version__)

        exitcode = 99
        try:
            self._write_job_for_command()
            exitcode = 0
        finally:
            logutils.write_task_finish_json(outdir=self._outdir,
                                            start_time=self._start_time,
                                            status=exitcode)
        return exitcode


class CellmapsvnnRunner(VnnRunner):
    """
    Class to run algorithm
    """

    def __init__(self, outdir=None,
                 command=None,
                 inputdir=None,
                 name=None,
                 organization_name=None,
                 project_name=None,
                 exitcode=None,
                 skip_logging=True,
                 input_data_dict=None,
                 provenance_utils=ProvenanceUtil()):
        """
        Constructor

        :param outdir: Directory to create and put results in
        :type outdir: str
        :param skip_logging: If ``True`` skip logging, if ``None`` or ``False`` do NOT skip logging
        :type skip_logging: bool
        :param exitcode: value to return via :py:meth:`.CellmapsvnnRunner.run` method
        :type int:
        :param input_data_dict: Command line arguments used to invoke this
        :type input_data_dict: dict
        :param provenance_utils: Wrapper for `fairscape-cli <https://pypi.org/project/fairscape-cli>`__
                                 which is used for
                                 `RO-Crate <https://www.researchobject.org/ro-crate>`__ creation and population
        :type provenance_utils: :py:class:`~cellmaps_utils.provenance.ProvenanceUtil`
        """
        super().__init__(outdir)
        self._command = command
        self._inputdir = inputdir
        self._name = name
        self._project_name = project_name
        self._organization_name = organization_name
        self._keywords = None
        self._description = None
        self._exitcode = exitcode
        self._start_time = int(time.time())
        if skip_logging is None:
            self._skip_logging = False
        else:
            self._skip_logging = skip_logging
        self._input_data_dict = input_data_dict
        self._provenance_utils = provenance_utils

        logger.debug('In constructor')

    def _update_provenance_fields(self):
        """
        Updates the provenance attributes by merging the ROCrate provenance attributes
        from the input directory with optional overrides for the name, project name, and organization name
        and additional keywords.
        """
        rocrate_dirs = []
        dirs = []
        if isinstance(self._inputdir, str):
            dirs = [self._inputdir]
        elif isinstance(self._inputdir, list):
            dirs = self._inputdir
        for entry in dirs:
            if os.path.exists(os.path.join(entry, constants.RO_CRATE_METADATA_FILE)):
                rocrate_dirs.append(entry)
        if len(rocrate_dirs) > 0:
            prov_attrs = self._provenance_utils.get_merged_rocrate_provenance_attrs(rocrate_dirs,
                                                                                    override_name=self._name,
                                                                                    override_project_name=
                                                                                    self._project_name,
                                                                                    override_organization_name=
                                                                                    self._organization_name,
                                                                                    extra_keywords=[
                                                                                        'VNN',
                                                                                        'Visible Neural Network',
                                                                                        self._command.COMMAND
                                                                                    ])
            if self._name is None:
                self._name = prov_attrs.get_name()

            if self._organization_name is None:
                self._organization_name = prov_attrs.get_organization_name()

            if self._project_name is None:
                self._project_name = prov_attrs.get_project_name()
            self._keywords = prov_attrs.get_keywords()
            self._description = prov_attrs.get_description()
        else:
            self._name = 'VNN tool'
            self._organization_name = 'Example'
            self._project_name = 'Example'
            self._keywords = ['vnn']
            self._description = 'Example input dataset VNN'

    def _create_rocrate(self):
        """
        Creates rocrate for output directory

        :raises CellMapsProvenanceError: If there is an error
        """
        logger.debug('Registering rocrate with FAIRSCAPE')

        try:
            self._provenance_utils.register_rocrate(self._outdir,
                                                    name=self._name,
                                                    organization_name=self._organization_name,
                                                    project_name=self._project_name,
                                                    description=self._description,
                                                    keywords=self._keywords)
        except TypeError as te:
            raise CellmapsvnnError('Invalid provenance: ' + str(te))
        except KeyError as ke:
            raise CellmapsvnnError('Key missing in provenance: ' + str(ke))

    def _register_software(self):
        """
        Registers this tool

        :raises CellMapsImageEmbeddingError: If fairscape call fails
        """
        software_keywords = self._keywords
        software_keywords.extend(['tools', cellmaps_vnn.__name__])
        software_description = self._description + ' ' + cellmaps_vnn.__description__
        self._softwareid = self._provenance_utils.register_software(self._outdir,
                                                                    name=cellmaps_vnn.__name__,
                                                                    description=software_description,
                                                                    author=cellmaps_vnn.__author__,
                                                                    version=cellmaps_vnn.__version__,
                                                                    file_format='py',
                                                                    keywords=software_keywords,
                                                                    url=cellmaps_vnn.__repo_url__)

    def _register_computation(self, generated_dataset_ids=[]):
        """
        Registers the computation run details with the FAIRSCAPE platform.

        :param generated_dataset_ids: List of IDs for the datasets generated during the computation.
        :type generated_dataset_ids: list
        # Todo: added in used dataset, software and what is being generated
        :return:
        """
        logger.debug('Getting id of input rocrate')
        used_dataset = []
        if isinstance(self._inputdir, str):
            if os.path.exists(os.path.join(self._inputdir, constants.RO_CRATE_METADATA_FILE)):
                used_dataset = [self._provenance_utils.get_id_of_rocrate(self._inputdir)]
        elif isinstance(self._inputdir, list):
            for entry in self._inputdir:
                if os.path.exists(os.path.join(entry, constants.RO_CRATE_METADATA_FILE)):
                    used_dataset.append(self._provenance_utils.get_id_of_rocrate(entry))

        self._provenance_utils.register_computation(self._outdir,
                                                    name=cellmaps_vnn.__name__ + ' computation',
                                                    run_by=str(self._provenance_utils.get_login()),
                                                    command=str(self._input_data_dict),
                                                    description='run of ' + cellmaps_vnn.__name__,
                                                    used_software=[self._softwareid],
                                                    used_dataset=used_dataset,
                                                    generated=generated_dataset_ids)

    def run(self):
        """
        Runs cellmaps_vnn

        :return:
        """
        exitcode = 99
        try:
            logger.debug('In run method')
            if os.path.isdir(self._outdir):
                raise CellmapsvnnError(self._outdir + ' already exists')
            if not os.path.isdir(self._outdir):
                os.makedirs(self._outdir, mode=0o755)
            if self._skip_logging is False:
                logutils.setup_filelogger(outdir=self._outdir,
                                          handlerprefix='cellmaps_vnn')
            logutils.write_task_start_json(outdir=self._outdir,
                                           start_time=self._start_time,
                                           data={'commandlineargs': self._input_data_dict},
                                           version=cellmaps_vnn.__version__)
            self._update_provenance_fields()

            self._create_rocrate()

            self._register_software()

            generated_dataset_ids = []

            if self._command:
                self._command.run()
                generated_dataset_ids.extend(self._command.register_outputs(
                    self._outdir, self._description, self._keywords, self._provenance_utils))
            else:
                raise CellmapsvnnError("No command provided to CellmapsvnnRunner")

            # register generated datasets
            self._register_computation(generated_dataset_ids=generated_dataset_ids)
            # set exit code to value passed in via constructor
            exitcode = self._exitcode if self._exitcode is not None else 0
        finally:
            # write a task finish file
            logutils.write_task_finish_json(outdir=self._outdir,
                                            start_time=self._start_time,
                                            status=exitcode)

        return exitcode
