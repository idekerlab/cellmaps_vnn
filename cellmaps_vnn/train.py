# train.py
import os
import shutil
from datetime import date

from cellmaps_utils import constants as constants
import cellmaps_vnn.constants as vnnconstants
import logging

import cellmaps_vnn
from cellmaps_vnn.data_wrapper import TrainingDataWrapper
from cellmaps_vnn.exceptions import CellmapsvnnError
from cellmaps_vnn.vnn_trainer import VNNTrainer
from cellmaps_vnn.util import copy_and_register_gene2id_file
from cellmaps_vnn.optuna_vnn_trainer import OptunaVNNTrainer

logger = logging.getLogger(__name__)


class VNNTrain:
    COMMAND = 'train'

    DEFAULT_EPOCH = 50
    DEFAULT_LR = 0.001
    DEFAULT_WD = 0.001
    DEFAULT_ALPHA = 0.3
    DEFAULT_PATIENCE = 30
    DEFAULT_DELTA = 0.001
    DEFAULT_MIN_DROPOUT_LAYER = 2
    DEFAULT_DROPOUT_FRACTION = 0.3
    DEFAULT_OPTIMIZE = 0
    DEFAULT_N_TRIALS = 3
    DEFAULT_STD = 'std.txt'

    def __init__(self, outdir, inputdir, gene_attribute_name=vnnconstants.GENE_SET_COLUMN_NAME, config_file=None,
                 training_data=None, gene2id=None, cell2id=None, mutations=None, cn_deletions=None,
                 cn_amplifications=None, batchsize=vnnconstants.DEFAULT_BATCHSIZE,
                 zscore_method=vnnconstants.DEFAULT_ZSCORE_METHOD, epoch=DEFAULT_EPOCH, lr=DEFAULT_LR, wd=DEFAULT_WD,
                 alpha=DEFAULT_ALPHA, genotype_hiddens=vnnconstants.DEFAULT_GENOTYPE_HIDDENS, patience=DEFAULT_PATIENCE,
                 delta=DEFAULT_DELTA, min_dropout_layer=DEFAULT_MIN_DROPOUT_LAYER, dropout_fraction=DEFAULT_DROPOUT_FRACTION,
                 optimize=DEFAULT_OPTIMIZE, n_trials=DEFAULT_N_TRIALS, cuda=vnnconstants.DEFAULT_CUDA,
                 skip_parent_copy=False, slurm=False, use_gpu=False, slurm_partition=None, slurm_account=None,
                 hierarchy=None, parent_network=None):
        """
        Constructor for training a Visual Neural Network.

        :param outdir: Directory to write results to.
        :type outdir: str
        :param inputdir: Path to directory or RO-Crate with hierarchy.cx2 file.
        :type inputdir: str
        :param gene_attribute_name: Name of the node attribute with genes/proteins.
        :type gene_attribute_name: str
        :param config_file: Path to configuration file for populating arguments.
        :type config_file: str, optional
        :param training_data: Training data file path.
        :type training_data: str, optional
        :param gene2id: File mapping genes to IDs.
        :type gene2id: str, optional
        :param cell2id: File mapping cells to IDs.
        :type cell2id: str, optional
        :param mutations: File with mutation information for cell lines.
        :type mutations: str, optional
        :param cn_deletions: File with copy number deletions for cell lines.
        :type cn_deletions: str, optional
        :param cn_amplifications: File with copy number amplifications for cell lines.
        :type cn_amplifications: str, optional
        :param batchsize: Batch size for training. Default is 64.
        :type batchsize: int
        :param zscore_method: Z-score method. Default is 'auc'.
        :type zscore_method: str
        :param epoch: Number of epochs for training. Default is 50.
        :type epoch: int
        :param lr: Learning rate. Default is 0.001.
        :type lr: float or list or tuple
        :param wd: Weight decay. Default is 0.001.
        :type wd: float
        :param alpha: Loss parameter alpha. Default is 0.3.
        :type alpha: float
        :param genotype_hiddens: Number of neurons in genotype parts. Default is 4.
        :type genotype_hiddens: int
        :param patience: Early stopping epoch limit. Default is 30.
        :type patience: int
        :param delta: Minimum loss improvement for early stopping. Default is 0.001.
        :type delta: float
        :param min_dropout_layer: Layer number to start applying dropout. Default is 2.
        :type min_dropout_layer: int
        :param dropout_fraction: Dropout fraction. Default is 0.3.
        :type dropout_fraction: float
        :param optimize: Hyperparameter optimization flag. Default is 0.
        :type optimize: int
        :param cuda: GPU index. Default is 0.
        :type cuda: int
        :param skip_parent_copy: If True, do not copy hierarchy parent. Default is False.
        :type skip_parent_copy: bool
        :param slurm: If True, generate SLURM script for training. Default is False.
        :type slurm: bool
        :param use_gpu: If True, adjust SLURM script to run on GPU. Default is False.
        :type use_gpu: bool
        :param slurm_partition: SLURM partition to use. Default is 'nrnb-gpu' if use_gpu is True.
        :type slurm_partition: str, optional
        :param slurm_account: SLURM account name.
        :type slurm_account: str, optional
        """
        self._outdir = os.path.abspath(outdir)
        self._inputdir = inputdir
        self._hierarchy = hierarchy
        self._parent_network = parent_network
        self._gene_attribute_name = gene_attribute_name
        self._config_file = config_file
        self._training_data = training_data
        self._gene2id = gene2id
        self._cell2id = cell2id
        self._mutations = mutations
        self._cn_deletions = cn_deletions
        self._cn_amplifications = cn_amplifications
        self._zscore_method = zscore_method
        self._epoch = epoch

        self._optimize = optimize
        self._batchsize, self._optimize_batchsize = self._process_param(batchsize, 'batchsize')
        self._lr, self._optimize_lr = self._process_param(lr, 'lr')
        self._wd, self._optimize_wd = self._process_param(wd, 'wd')
        self._alpha, self._optimize_alpha = self._process_param(alpha, 'alpha')
        self._genotype_hiddens, self._optimize_genotype_hiddens = self._process_param(genotype_hiddens,
                                                                                      'genotype_hiddens')
        self._patience, self._optimize_patience = self._process_param(patience, 'patience')
        self._delta, self._optimize_delta = self._process_param(delta, 'delta')
        self._min_dropout_layer, self._optimize_min_dropout_layer = self._process_param(min_dropout_layer,
                                                                                        'min_dropout_layer')
        self._dropout_fraction, self._optimize_dropout_fraction = self._process_param(dropout_fraction,
                                                                                      'dropout_fraction')

        self._n_trials = n_trials
        self._cuda = cuda
        self._skip_parent_copy = skip_parent_copy
        self._slurm = slurm
        self._use_gpu = use_gpu
        self._slurm_partition = slurm_partition
        self._slurm_account = slurm_account
        self._modelfile = self._get_model_dest_file()
        self._stdfile = self._get_std_dest_file()

    def _process_param(self, param, name):
        if isinstance(param, list) or isinstance(param, tuple):
            if isinstance(param, tuple) or len(param) > 1:
                if self._optimize != 0:
                    return param[0], param
                else:
                    raise CellmapsvnnError(
                        f'Hyperparameter optimization is not enabled (optimize set to 0), but multiple values '
                        f'for {name} are provided. Please specify only one value, or enable optimization.'
                    )
            else:
                return param[0], None
        else:
            return param, None

    @staticmethod
    def add_subparser(subparsers):
        """
        Adds a subparser for the 'train' command.
        """
        # TODO: modify description later
        desc = """
        Version: todo

        The 'train' command trains a Visual Neural Network using specified hierarchy files
        and data from drugcell or NeSTVNN repository. The resulting model is stored in a
        directory specified by the user.
        """
        parser = subparsers.add_parser(VNNTrain.COMMAND,
                                       help='Train a Visual Neural Network',
                                       description=desc,
                                       formatter_class=constants.ArgParseFormatter)
        parser.add_argument('outdir', help='Directory to write results to')
        parser.add_argument('--inputdir', help='Path to directory or RO-Crate with hierarchy.cx2 file.'
                                               'Note that the name of the hierarchy should be hierarchy.cx2.')
        parser.add_argument('--hierarchy', help='Path to hierarchy (optional). If not set, the process will search for '
                                                'hierarchy.cx2 in inputdir. It will fail if not found.', type=str)
        parser.add_argument('--parent_network', help='Path to interactome (parent network, optional) of hierarchy '
                                                     'or NDEx UUID of parent network. If not set, the process will '
                                                     'search for hierarchy_parent.cx2 in inputdir, but it will not fail'
                                                     'if not found.', type=str)
        parser.add_argument('--gene_attribute_name', help='Name of the node attribute of the hierarchy with list of '
                                                          'genes/ proteins of this node. Default: CD_MemberList.',
                            type=str, default=vnnconstants.GENE_SET_COLUMN_NAME)
        parser.add_argument('--config_file', help='Config file that can be used to populate arguments for training. '
                                                  'If a given argument is set, it will override the default value.')
        parser.add_argument('--training_data', help='Training data')
        parser.add_argument('--gene2id', help='Gene to ID mapping file', type=str)
        parser.add_argument('--cell2id', help='Cell to ID mapping file', type=str)
        parser.add_argument('--mutations', help='Mutation information for cell lines', type=str)
        parser.add_argument('--cn_deletions', help='Copy number deletions for cell lines', type=str)
        parser.add_argument('--cn_amplifications', help='Copy number amplifications for cell lines',
                            type=str)
        parser.add_argument('--epoch', type=int, default=VNNTrain.DEFAULT_EPOCH, help='Training epochs')
        parser.add_argument('--zscore_method', type=str, choices=['auc', 'zscore', 'robustz'],
                            help='Z-score method (choose from: auc, zscore, robustz). Default: auc')
        parser.add_argument('--batchsize', type=int, nargs='+', help='Batch size')
        parser.add_argument('--lr', type=float, nargs='+', help='Learning rate')
        parser.add_argument('--wd', type=float, nargs='+', help='Weight decay')
        parser.add_argument('--alpha', type=float, nargs='+', help='Loss parameter alpha')
        parser.add_argument('--genotype_hiddens', type=int, nargs='+', help='Neurons in genotype parts')
        parser.add_argument('--patience', type=int, nargs='+', help='Early stopping epoch limit')
        parser.add_argument('--delta', type=float, nargs='+', help='Minimum change in loss for improvement')
        parser.add_argument('--min_dropout_layer', type=int, nargs='+', help='Start dropout from this layer')
        parser.add_argument('--dropout_fraction', type=float, nargs='+', help='Dropout fraction')
        parser.add_argument('--optimize', type=int, help='Hyperparameter optimization (0- no optimization, '
                                                         '1- optuna optimizer)')
        parser.add_argument('--n_trials', type=int, help='Number of trials in hyperparameter optimization')
        parser.add_argument('--cuda', type=int, help='Specify GPU')
        parser.add_argument('--skip_parent_copy', help='If set, hierarchy parent (interactome) will not be copied',
                            action='store_true')
        parser.add_argument('--slurm', help='If set, slurm script for training will be generated.',
                            action='store_true')
        parser.add_argument('--use_gpu', help='If set, slurm script will be adjusted to run on GPU.',
                            action='store_true')
        parser.add_argument('--slurm_partition', help='Slurm partition. If use_gpu is set, the default is nrnb-gpu.',
                            type=str)
        parser.add_argument('--slurm_account', help='Slurm account. If use_gpu is set, the default is nrnb-gpu.',
                            type=str)
        return parser

    def run(self):
        """
        The logic for training the Visual Neural Network.
        """
        if self._optimize not in [0, 1]:
            logger.error(f"The value {self._optimize} is wrong value for optimize.")
            raise CellmapsvnnError(f"The value {self._optimize} is wrong value for optimize.")

        data_wrapper = TrainingDataWrapper(self._outdir, self._inputdir, self._gene_attribute_name,
                                           self._training_data, self._cell2id, self._gene2id, self._mutations,
                                           self._cn_deletions, self._cn_amplifications, self._modelfile,
                                           self._genotype_hiddens, self._lr, self._wd, self._alpha, self._epoch,
                                           self._batchsize, self._cuda, self._zscore_method, self._stdfile,
                                           self._patience, self._delta, self._min_dropout_layer,
                                           self._dropout_fraction, self._hierarchy)
        if self._optimize == 1:
            trial_params = OptunaVNNTrainer(data_wrapper,
                                            n_trials=self._n_trials,
                                            batchsize_vals=self._optimize_batchsize,
                                            lr_vals=self._optimize_lr,
                                            wd_vals=self._optimize_wd,
                                            alpha_vals=self._optimize_alpha,
                                            genotype_hiddens_vals=self._optimize_genotype_hiddens,
                                            patience_vals=self._optimize_patience,
                                            delta_vals=self._optimize_delta,
                                            min_dropout_layer_vals=self._optimize_min_dropout_layer,
                                            dropout_fraction_vals=self._optimize_dropout_fraction
                                            ).exec_study()
            for key, value in trial_params.items():
                if hasattr(data_wrapper, key):
                    setattr(data_wrapper, key, value)
        try:
            VNNTrainer(data_wrapper).train_model()
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise CellmapsvnnError(f"Encountered problem in training: {e}")

    def _get_model_dest_file(self):
        """
        Returns the file path for saving the trained model file.

        :return: The file path for the model file.
        """
        return os.path.join(self._outdir, 'model_final.pt')

    def _get_std_dest_file(self):
        """
        Returns the file path for saving the standard deviation file.

        :return: The file path for the standard deviation file.
        """
        return os.path.join(self._outdir, VNNTrain.DEFAULT_STD)

    def register_outputs(self, outdir, description, keywords, provenance_utils):
        """
        Registers the model and standard deviation files with the FAIRSCAPE service for data provenance.
        It generates dataset IDs for each registered file.

        :param outdir: The directory where the output files are stored.
        :param description: Description for the output files.
        :param keywords: List of keywords associated with the files.
        :param provenance_utils: The utility class for provenance registration.

        :return: A list of dataset IDs for the registered model and standard deviation files.
        """
        return_ids = [self._register_model_file(outdir, description, keywords, provenance_utils),
                      self._register_std_file(outdir, description, keywords, provenance_utils),
                      self._copy_and_register_hierarchy(outdir, description, keywords, provenance_utils),
                      self._register_pruned_hierarchy(outdir, description, keywords, provenance_utils),
                      copy_and_register_gene2id_file(self._gene2id, outdir, description, keywords,
                                                     provenance_utils)]
        if not self._skip_parent_copy:
            id_hierarchy_parent = self._copy_and_register_hierarchy_parent(outdir, description, keywords,
                                                                           provenance_utils)
            if id_hierarchy_parent is not None:
                return_ids.append(id_hierarchy_parent)

        return return_ids

    def _register_model_file(self, outdir, description, keywords, provenance_utils):
        """
        Registers the trained model file with the FAIRSCAPE service for data provenance.

        :param outdir: The output directory where the model file is stored.
        :param description: Description of the model file for provenance registration.
        :param keywords: List of keywords associated with the model file.
        :param provenance_utils: The utility class for provenance registration.

        :return: The dataset ID assigned to the registered model file.
        """
        dest_path = self._get_model_dest_file()
        description = description
        description += ' Model file'
        keywords = keywords
        keywords.extend(['file'])
        data_dict = {'name': os.path.basename(dest_path) + ' trained model file',
                     'description': description,
                     'keywords': keywords,
                     'data-format': 'pt',
                     'author': cellmaps_vnn.__name__,
                     'version': cellmaps_vnn.__version__,
                     'date-published': date.today().strftime(provenance_utils.get_default_date_format_str())}
        dataset_id = provenance_utils.register_dataset(outdir,
                                                       source_file=dest_path,
                                                       data_dict=data_dict)
        return dataset_id

    def _register_std_file(self, outdir, description, keywords, provenance_utils):
        """
        Registers the standard deviation file with the FAIRSCAPE service for data provenance.

        :param outdir: The output directory where the standard deviation file is stored.
        :param description: Description of the standard deviation file for provenance registration.
        :param keywords: List of keywords associated with the standard deviation file.
        :param provenance_utils: The utility class for provenance registration.

        :return: The dataset ID assigned to the registered standard deviation file.
        """
        dest_path = self._get_std_dest_file()
        description = description
        description += ' standard deviation file'
        keywords = keywords
        keywords.extend(['file'])
        data_dict = {'name': os.path.basename(dest_path) + ' standard deviation file',
                     'description': description,
                     'keywords': keywords,
                     'data-format': 'txt',
                     'author': cellmaps_vnn.__name__,
                     'version': cellmaps_vnn.__version__,
                     'date-published': date.today().strftime(provenance_utils.get_default_date_format_str())}
        dataset_id = provenance_utils.register_dataset(outdir,
                                                       source_file=dest_path,
                                                       data_dict=data_dict)
        return dataset_id

    def _copy_and_register_hierarchy(self, outdir, description, keywords, provenance_utils):
        hierarchy_out_file = os.path.join(outdir, vnnconstants.ORIGINAL_HIERARCHY_FILENAME)
        hierarchy_in_file = os.path.join(self._inputdir, vnnconstants.HIERARCHY_FILENAME) if self._hierarchy is None \
            else self._hierarchy
        shutil.copy(hierarchy_in_file, hierarchy_out_file)

        data_dict = {'name': os.path.basename(hierarchy_out_file) + ' Hierarchy network file',
                     'description': description + ' Hierarchy network file',
                     'keywords': keywords,
                     'data-format': 'CX2',
                     'author': cellmaps_vnn.__name__,
                     'version': cellmaps_vnn.__version__,
                     'date-published': date.today().strftime('%m-%d-%Y')}
        dataset_id = provenance_utils.register_dataset(outdir,
                                                       source_file=hierarchy_out_file,
                                                       data_dict=data_dict)
        return dataset_id

    def _register_pruned_hierarchy(self, outdir, description, keywords, provenance_utils):
        hierarchy_out_file = os.path.join(outdir, vnnconstants.HIERARCHY_FILENAME)

        data_dict = {'name': os.path.basename(hierarchy_out_file) + ' Hierarchy network file used to build VNN',
                     'description': description + ' Hierarchy network file used to build VNN',
                     'keywords': keywords,
                     'data-format': 'CX2',
                     'author': cellmaps_vnn.__name__,
                     'version': cellmaps_vnn.__version__,
                     'date-published': date.today().strftime('%m-%d-%Y')}
        dataset_id = provenance_utils.register_dataset(outdir,
                                                       source_file=hierarchy_out_file,
                                                       data_dict=data_dict)
        return dataset_id

    def _copy_and_register_hierarchy_parent(self, outdir, description, keywords, provenance_utils):
        hierarchy_parent_in_file = None
        if self._parent_network is not None:
            hierarchy_parent_in_file = self._parent_network
        if self._inputdir is not None:
            hierarchy_parent_in_file = os.path.join(self._inputdir, vnnconstants.PARENT_NETWORK_NAME)
        if hierarchy_parent_in_file is None or not os.path.exists(hierarchy_parent_in_file):
            logger.warning("No hierarchy parent in the input directory. Cannot copy.")
            return None
        hierarchy_parent_out_file = os.path.join(outdir, vnnconstants.PARENT_NETWORK_NAME)
        shutil.copy(hierarchy_parent_in_file, hierarchy_parent_out_file)

        data_dict = {'name': os.path.basename(hierarchy_parent_out_file) + ' Hierarchy parent network file',
                     'description': description + ' Hierarchy parent network file',
                     'keywords': keywords,
                     'data-format': 'CX2',
                     'author': cellmaps_vnn.__name__,
                     'version': cellmaps_vnn.__version__,
                     'date-published': date.today().strftime('%m-%d-%Y')}
        dataset_id = provenance_utils.register_dataset(outdir,
                                                       source_file=hierarchy_parent_out_file,
                                                       data_dict=data_dict)
        return dataset_id
