# train.py
from cellmaps_utils import constants

from cellmaps_vnn.data_wrapper import DataWrapper


class VNNTrain:
    COMMAND = 'train'

    def __init__(self, theargs):
        """
        Constructor for training Visual Neural Network.
        """
        self._theargs = theargs

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
        parser.add_argument('--hierarchy', required=True, help='Path to hierarchy.cx2 file in RO-Crate')
        parser.add_argument('--hierarchy_parent', required=True, help='Path to hierarchy_parent.cx2 file in RO-Crate')
        parser.add_argument('--training_data', required=True, help='Training data')
        # removed: onto (replaced with hierarchy), train (replaced with training_data)
        parser.add_argument('--gene2id', help='Gene to ID mapping file', type=str)
        parser.add_argument('--cell2id', help='Cell to ID mapping file', type=str)
        parser.add_argument('--mutations', help='Mutation information for cell lines', type=str)
        parser.add_argument('--cn_deletions', help='Copy number deletions for cell lines', type=str)
        parser.add_argument('--cn_amplifications', help='Copy number amplifications for cell lines', type=str)
        # TODO: verify above arguments
        parser.add_argument('--epoch', help='Training epochs for training', type=int, default=300)
        parser.add_argument('--lr', help='Learning rate', type=float, default=0.001)
        parser.add_argument('--wd', help='Weight decay', type=float, default=0.001)
        parser.add_argument('--alpha', help='Loss parameter alpha', type=float, default=0.3)
        parser.add_argument('--batchsize', help='Batchsize', type=int, default=64)
        parser.add_argument('--modeldir', help='Folder for trained models', type=str, default='MODEL/')
        parser.add_argument('--cuda', help='Specify GPU', type=int, default=0)
        parser.add_argument('--genotype_hiddens',
                            help='Mapping for the number of neurons in each term in genotype parts',
                            type=int, default=4)
        parser.add_argument('--optimize', help='Hyper-parameter optimization', type=int, default=1)
        parser.add_argument('--zscore_method', help='zscore method (zscore/robustz)', type=str, default='auc')
        parser.add_argument('--std', help='Standardization File', type=str, default='MODEL/std.txt')
        parser.add_argument('--patience', help='Early stopping epoch limit', type=int, default=30)
        parser.add_argument('--delta', help='Minimum change in loss to be considered an improvement', type=float,
                            default=0.001)
        parser.add_argument('--min_dropout_layer', help='Start dropout from this Layer number', type=int, default=2)
        parser.add_argument('--dropout_fraction', help='Dropout Fraction', type=float, default=0.3)

        # TODO: Add other necessary arguments
        return parser

    def run(self):
        """
        The logic for training the Visual Neural Network.
        """
        # TODO: Implement training logic
        data_wrapper = DataWrapper(self._theargs)
        pass
