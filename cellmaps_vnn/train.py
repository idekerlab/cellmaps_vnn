# train.py
import os
from datetime import date

from cellmaps_utils import constants
import logging

import cellmaps_vnn
from cellmaps_vnn.data_wrapper import TrainingDataWrapper
from cellmaps_vnn.exceptions import CellmapsvnnError
from cellmaps_vnn.vnn_trainer import VNNTrainer

logger = logging.getLogger(__name__)


class VNNTrain:
    COMMAND = 'train'

    def __init__(self, theargs):
        """
        Constructor for training Visual Neural Network.
        """
        self._theargs = theargs
        self._theargs.modelfile = self._get_model_dest_file()
        self._theargs.stdfile = self._get_std_dest_file()

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
        parser.add_argument('--inputdir', required=True, help='Path to RO-Crate with hierarchy.cx2 file.')
        parser.add_argument('--training_data', required=True, help='Training data')
        parser.add_argument('--gene2id', required=True, help='Gene to ID mapping file', type=str)
        parser.add_argument('--cell2id', required=True, help='Cell to ID mapping file', type=str)
        parser.add_argument('--mutations', required=True, help='Mutation information for cell lines', type=str)
        parser.add_argument('--cn_deletions', required=True, help='Copy number deletions for cell lines', type=str)
        parser.add_argument('--cn_amplifications', required=True, help='Copy number amplifications for cell lines',
                            type=str)
        parser.add_argument('--batchsize', help='Batchsize', type=int, default=64)
        parser.add_argument('--cuda', help='Specify GPU', type=int, default=0)
        parser.add_argument('--zscore_method', help='zscore method (zscore/robustz)', type=str, default='auc')
        parser.add_argument('--std', help='Standardization File name', type=str, default='std.txt')
        parser.add_argument('--epoch', help='Training epochs for training', type=int, default=300)
        parser.add_argument('--lr', help='Learning rate', type=float, default=0.001)
        parser.add_argument('--wd', help='Weight decay', type=float, default=0.001)
        parser.add_argument('--alpha', help='Loss parameter alpha', type=float, default=0.3)
        parser.add_argument('--genotype_hiddens',
                            help='Mapping for the number of neurons in each term in genotype parts',
                            type=int, default=4)
        parser.add_argument('--optimize', help='Hyper-parameter optimization', type=int, default=1)
        parser.add_argument('--patience', help='Early stopping epoch limit', type=int, default=30)
        parser.add_argument('--delta', help='Minimum change in loss to be considered an improvement', type=float,
                            default=0.001)
        parser.add_argument('--min_dropout_layer', help='Start dropout from this Layer number', type=int, default=2)
        parser.add_argument('--dropout_fraction', help='Dropout Fraction', type=float, default=0.3)
        return parser

    def run(self):
        """
        The logic for training the Visual Neural Network.
        """
        try:
            data_wrapper = TrainingDataWrapper(self._theargs)
            if self._theargs.optimize == 1:
                VNNTrainer(data_wrapper).train_model()
            else:
                logger.error(f"The value {self._theargs.optimize} is wrong value for optimize.")
                raise CellmapsvnnError(f"The value {self._theargs.optimize} is wrong value for optimize.")
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise CellmapsvnnError(f"Encountered problem in training: {e}")

    def _get_model_dest_file(self):
        return os.path.join(self._theargs.outdir, 'model_final.pt')

    def _get_std_dest_file(self):
        return os.path.join(self._theargs.outdir, self._theargs.std)

    def register_outputs(self, outdir, description, keywords, provenance_utils):
        id_model = self._register_model_file(outdir, description, keywords, provenance_utils)
        id_std = self._register_std_file(outdir, description, keywords, provenance_utils)
        return [id_model,  id_std]

    def _register_model_file(self, outdir, description, keywords, provenance_utils):
        """
        TODO

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
        TODO

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

