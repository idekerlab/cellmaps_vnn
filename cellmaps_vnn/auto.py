import logging
import os

import yaml
from cellmaps_utils import constants as constants

from cellmaps_vnn.annotate import VNNAnnotate
from cellmaps_vnn.exceptions import CellmapsvnnError
from cellmaps_vnn.predict import VNNPredict
from cellmaps_vnn.runner import SLURMCellmapsvnnRunner
from cellmaps_vnn.train import VNNTrain

logger = logging.getLogger(__name__)


class VNNAuto:
    COMMAND = 'auto'

    def __init__(self, outdir=None, config_file=None, data_dir=None, hierarchy_dir=None, no_of_models=1,
                 use_gpu=False, slurm=False, slurm_partition=None, slurm_account=None):
        self._outdir = outdir
        self._config_file = config_file
        self._data_dir = data_dir
        self._hierarchy_dir = hierarchy_dir
        self._no_of_models = no_of_models
        self._use_gpu = use_gpu
        self._slurm = slurm
        self._slurm_partition = slurm_partition
        self._slurm_account = slurm_account

    @staticmethod
    def add_subparser(subparsers):
        """
        Adds a subparser for the 'auto' command.
        """
        desc = """
        Version: todo

        The 'auto' command runs the complete pipeline of the Cell Maps Visual Neural Network packages,
        specified number of times, it takes a directory with input feature data and hierarchy.
        The results, model and annotated hierarchy are stored in a directory specified by the user.
        """
        parser = subparsers.add_parser(VNNAuto.COMMAND,
                                       help='Train a Visual Neural Network',
                                       description=desc,
                                       formatter_class=constants.ArgParseFormatter)
        parser.add_argument('outdir', help='Directory to write results to')
        parser.add_argument('--config_file', required=True, help='Path to parameterized config file.')
        parser.add_argument('--data_dir', required=True, help='Path to directory with data files.')
        parser.add_argument('--hierarchy_dir', help="Specifies the path to hierarchy if is not located in data_dir "
                                                    "or is located in separate RO-Crate.")
        parser.add_argument('--no_of_models', type=int, default=1, help='Number of models to be generated.')
        parser.add_argument('--use_gpu', help='If set, slurm script will be adjusted to run on GPU.',
                            action='store_true')
        parser.add_argument('--slurm', help='If set, slurm script for training will be generated.',
                            action='store_true')
        parser.add_argument('--slurm_partition', help='Slurm partition.', type=str)
        parser.add_argument('--slurm_account', help='Slurm account.', type=str)

    def run(self):
        if self._slurm is False:
            raise CellmapsvnnError("Not implemented to run without slurm!")

        config_dicts = self._generate_config()
        for i in range(1, self._no_of_models + 1):
            model_outdir = f"{self._outdir}/model_{i}"
            os.makedirs(model_outdir, exist_ok=True)
            self.generate_slurm_for_model(model_outdir, config_dicts[i - 1])

    def _generate_config(self):
        """
        Generate multiple config files by replacing placeholders with numbers [1, no_of_models]
        """
        config_dicts = []
        for i in range(1, self._no_of_models + 1):
            with open(self._config_file, 'r') as f:
                config = yaml.safe_load(f)

            # Replace placeholders dynamically
            for key, value in config.items():
                if isinstance(value, str) and "{i}" in value:
                    config[key] = value.format(i=i)
            config_dicts.append(config)
        return config_dicts

    def generate_slurm_for_model(self, model_outdir, config_dict):
        slurm_partition = 'nrnb-gpu' if (self._slurm_partition is None and self._use_gpu) else self._slurm_partition
        slurm_account = 'nrnb-gpu' if (self._slurm_account is None and self._use_gpu) else self._slurm_account

        runner = SLURMCellmapsvnnRunner(outdir=model_outdir,
                                        inputdir=getattr(config_dict, 'inputdir', None),
                                        gene_attribute_name=getattr(config_dict, 'gene_attribute_name', None),
                                        gene2id=getattr(config_dict, 'gene2id', None),
                                        cell2id=getattr(config_dict, 'cell2id', None),
                                        mutations=getattr(config_dict, 'mutations', None),
                                        cn_deletions=getattr(config_dict, 'cn_deletions', None),
                                        cn_amplifications=getattr(config_dict, 'cn_amplifications', None),
                                        training_data=getattr(config_dict, 'training_data', None),
                                        batchsize=getattr(config_dict, 'batchsize', None),
                                        cuda=getattr(config_dict, 'cuda', None),
                                        zscore_method=getattr(config_dict, 'zscore_method', None),
                                        epoch=getattr(config_dict, 'epoch', None),
                                        lr=getattr(config_dict, 'lr', None),
                                        wd=getattr(config_dict, 'wd', None),
                                        alpha=getattr(config_dict, 'alpha', None),
                                        genotype_hiddens=getattr(config_dict, 'genotype_hiddens', None),
                                        optimize=getattr(config_dict, 'optimize', None),
                                        n_trials=getattr(config_dict, 'n_trials', None),
                                        patience=getattr(config_dict, 'patience', None),
                                        delta=getattr(config_dict, 'delta', None),
                                        min_dropout_layer=getattr(config_dict, 'min_dropout_layer', None),
                                        dropout_fraction=getattr(config_dict, 'dropout_fraction', None),
                                        skip_parent_copy=getattr(config_dict, 'skip_parent_copy', False),
                                        cpu_count=getattr(config_dict, 'cpu_count', None),
                                        drug_count=getattr(config_dict, 'drug_count', None),
                                        predict_data=getattr(config_dict, 'predict_data', None),
                                        std=getattr(config_dict, 'std', None),
                                        model_predictions=getattr(config_dict, 'model_predictions', None),
                                        disease=getattr(config_dict, 'disease', None),
                                        hierarchy=getattr(config_dict, 'hierarchy', None),
                                        parent_network=getattr(config_dict, 'parent_network', None),
                                        ndexserver=getattr(config_dict, 'ndexserver', None),
                                        ndexuser=getattr(config_dict, 'ndexuser', None),
                                        ndexpassword=getattr(config_dict, 'ndexpassword', None),
                                        visibility=getattr(config_dict, 'visibility', False),
                                        gpu=self._use_gpu,
                                        slurm_partition=slurm_partition,
                                        slurm_account=slurm_account
                                        )
        runner._command = VNNTrain
        runner.run()
        runner._command = VNNPredict
        runner.run()
        runner._command = VNNAnnotate
        runner.run()
