# train.py
import argparse


class VNNTrain:
    COMMAND = 'train'

    def __init__(self, theargs):
        """
        Constructor for training Visual Neural Network.
        """
        self._theargs = theargs
        # TODO: Initialize other necessary variables or configurations

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
                                       # TODO: choose formatter
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--hierarchy', help='Path to hierarchy.cx2 file in RO-Crate')
        parser.add_argument('--parent_hierarchy', help='Path to parent_hierarchy.cx2 file in RO-Crate')
        parser.add_argument('--training_data', help='Training data')
        # TODO: Add other necessary arguments
        return parser

    def run(self):
        """
        The logic for training the Visual Neural Network.
        """
        # TODO: Implement training logic
        pass
