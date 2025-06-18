#! /usr/bin/env python

import argparse
import sys
import logging
import logging.config

import yaml
from cellmaps_utils import logutils
from cellmaps_utils import constants
import cellmaps_vnn
from cellmaps_vnn.annotate import VNNAnnotate
from cellmaps_vnn.exceptions import CellmapsvnnError
from cellmaps_vnn.predict import VNNPredict
from cellmaps_vnn.runner import CellmapsvnnRunner, SLURMCellmapsvnnRunner
from cellmaps_vnn.train import VNNTrain
import cellmaps_vnn.constants as vnnconstants

logger = logging.getLogger(__name__)


def _parse_arguments(desc, args):
    """
    Parses command line arguments

    :param desc: description to display on command line
    :type desc: str
    :param args: command line arguments usually :py:func:`sys.argv[1:]`
    :type args: list
    :return: arguments parsed by :py:mod:`argparse`
    :rtype: :py:class:`argparse.Namespace`
    """
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=constants.ArgParseFormatter)
    subparsers = parser.add_subparsers(dest='command', help='Command to run. Type <command> -h for more help')
    subparsers.required = True

    VNNTrain.add_subparser(subparsers)
    VNNPredict.add_subparser(subparsers)
    VNNAnnotate.add_subparser(subparsers)
    parser.add_argument('--logconf', default=None,
                        help='Path to python logging configuration file in '
                             'this format: https://docs.python.org/3/library/'
                             'logging.config.html#logging-config-fileformat '
                             'Setting this overrides -v parameter which uses '
                             ' default logger. (default None)')
    parser.add_argument('--exitcode', help='Exit code this command will return',
                        default=0, type=int)
    parser.add_argument('--skip_logging', action='store_true',
                        help='If set, output.log, error.log '
                             'files will not be created')
    parser.add_argument('--provenance',
                        help='Path to file containing provenance '
                             'information about input files in JSON format. '
                             'This is required and not including will output '
                             'and error message with example of file')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                        help='Increases verbosity of logger to standard '
                             'error for log messages in this module. Messages are '
                             'output at these python logging levels '
                             '-v = WARNING, -vv = INFO, '
                             '-vvv = DEBUG, -vvvv = NOTSET (default ERROR '
                             'logging)')
    parser.add_argument('--version', action='version',
                        version=('%(prog)s ' +
                                 cellmaps_vnn.__version__))

    return parser.parse_args(args)


def main(args):
    """
    Main entry point for program

    :param args: arguments passed to command line usually :py:func:`sys.argv[1:]`
    :type args: list

    :return: return value of :py:meth:`cellmaps_vnn.runner.CellmapsvnnRunner.run`
             or ``2`` if an exception is raised
    :rtype: int
    """
    desc = """
    Version {version}

    Invokes run() method on CellmapsvnnRunner

    """.format(version=cellmaps_vnn.__version__)
    theargs = _parse_arguments(desc, args[1:])
    theargs.program = args[0]
    theargs.version = cellmaps_vnn.__version__

    config = {}
    if theargs.command == VNNTrain.COMMAND or theargs.command == VNNPredict.COMMAND:
        if theargs.config_file is not None:
            with open(theargs.config_file, "r") as file:
                config = yaml.safe_load(file)

    for key in vars(theargs):
        value = getattr(theargs, key)
        if value not in (None, False):
            config[key] = value

    if theargs.command == VNNTrain.COMMAND or theargs.command == VNNPredict.COMMAND:
        required_args = ['cell2id', 'mutations', 'cn_deletions', 'cn_amplifications']
        if theargs.command == VNNTrain.COMMAND:
            required_args.append('gene2id')
            required_args.append('training_data')
        else:
            required_args.append('predict_data')

        missing_args = []
        for arg in required_args:
            if getattr(theargs, arg, None) is None and arg not in config:
                missing_args.append(arg)

        if missing_args:
            raise CellmapsvnnError(
                f"The following arguments are required either in the command line "
                f"or config file: {', '.join(missing_args)}")

    set_arguments_from_config_and_defaults(theargs, config)

    try:
        logutils.setup_cmd_logging(theargs)

        if theargs.command == VNNTrain.COMMAND:
            cmd = VNNTrain(
                theargs.outdir,
                theargs.inputdir,
                theargs.gene_attribute_name,
                config_file=theargs.config_file,
                training_data=theargs.training_data,
                gene2id=theargs.gene2id,
                cell2id=theargs.cell2id,
                mutations=theargs.mutations,
                cn_deletions=theargs.cn_deletions,
                cn_amplifications=theargs.cn_amplifications,
                batchsize=theargs.batchsize,
                zscore_method=theargs.zscore_method,
                epoch=theargs.epoch,
                lr=theargs.lr,
                wd=theargs.wd,
                alpha=theargs.alpha,
                genotype_hiddens=theargs.genotype_hiddens,
                patience=theargs.patience,
                delta=theargs.delta,
                min_dropout_layer=theargs.min_dropout_layer,
                dropout_fraction=theargs.dropout_fraction,
                optimize=theargs.optimize,
                n_trials=theargs.n_trials,
                cuda=theargs.cuda,
                skip_parent_copy=theargs.skip_parent_copy,
                slurm=theargs.slurm,
                use_gpu=theargs.use_gpu,
                slurm_partition=theargs.slurm_partition,
                slurm_account=theargs.slurm_account,
                hierarchy=theargs.hierarchy,
                parent_network=theargs.parent_network
            )
        elif theargs.command == VNNPredict.COMMAND:
            cmd = VNNPredict(
                theargs.outdir,
                theargs.inputdir,
                config_file=theargs.config_file,
                predict_data=theargs.predict_data,
                gene2id=theargs.gene2id,
                cell2id=theargs.cell2id,
                mutations=theargs.mutations,
                cn_deletions=theargs.cn_deletions,
                cn_amplifications=theargs.cn_amplifications,
                batchsize=theargs.batchsize,
                zscore_method=theargs.zscore_method,
                cpu_count=theargs.cpu_count,
                drug_count=theargs.drug_count,
                genotype_hiddens=theargs.genotype_hiddens,
                std=theargs.std,
                cuda=theargs.cuda,
                slurm=theargs.slurm,
                use_gpu=theargs.use_gpu,
                slurm_partition=theargs.slurm_partition,
                slurm_account=theargs.slurm_account
            )
        elif theargs.command == VNNAnnotate.COMMAND:
            cmd = VNNAnnotate(
                theargs.outdir,
                theargs.model_predictions,
                disease=theargs.disease,
                hierarchy=theargs.hierarchy,
                parent_network=theargs.parent_network,
                ndexserver=theargs.ndexserver,
                ndexuser=theargs.ndexuser,
                ndexpassword=theargs.ndexpassword,
                visibility=theargs.visibility,
                slurm=theargs.slurm,
                slurm_partition=theargs.slurm_partition,
                slurm_account=theargs.slurm_account
            )
            theargs.inputdir = theargs.model_predictions
        else:
            raise Exception('Invalid command: ' + str(theargs.command))

        if theargs.slurm:
            use_gpu = True if (theargs.command != VNNAnnotate.COMMAND and theargs.use_gpu) else False
            slurm_partition = 'nrnb-gpu' if (theargs.slurm_partition is None and use_gpu) else theargs.slurm_partition
            slurm_account = 'nrnb-gpu' if (theargs.slurm_account is None and use_gpu) else theargs.slurm_account

            runner = SLURMCellmapsvnnRunner(outdir=theargs.outdir,
                                            command=cmd,
                                            inputdir=getattr(theargs, 'inputdir', None),
                                            gene_attribute_name=getattr(theargs, 'gene_attribute_name', None),
                                            gene2id=getattr(theargs, 'gene2id', None),
                                            cell2id=getattr(theargs, 'cell2id', None),
                                            mutations=getattr(theargs, 'mutations', None),
                                            cn_deletions=getattr(theargs, 'cn_deletions', None),
                                            cn_amplifications=getattr(theargs, 'cn_amplifications', None),
                                            training_data=getattr(theargs, 'training_data', None),
                                            batchsize=getattr(theargs, 'batchsize', None),
                                            cuda=getattr(theargs, 'cuda', None),
                                            zscore_method=getattr(theargs, 'zscore_method', None),
                                            epoch=getattr(theargs, 'epoch', None),
                                            lr=getattr(theargs, 'lr', None),
                                            wd=getattr(theargs, 'wd', None),
                                            alpha=getattr(theargs, 'alpha', None),
                                            genotype_hiddens=getattr(theargs, 'genotype_hiddens', None),
                                            optimize=getattr(theargs, 'optimize', None),
                                            n_trials=getattr(theargs, 'n_trials', None),
                                            patience=getattr(theargs, 'patience', None),
                                            delta=getattr(theargs, 'delta', None),
                                            min_dropout_layer=getattr(theargs, 'min_dropout_layer', None),
                                            dropout_fraction=getattr(theargs, 'dropout_fraction', None),
                                            skip_parent_copy=getattr(theargs, 'skip_parent_copy', False),
                                            cpu_count=getattr(theargs, 'cpu_count', None),
                                            drug_count=getattr(theargs, 'drug_count', None),
                                            predict_data=getattr(theargs, 'predict_data', None),
                                            std=getattr(theargs, 'std', None),
                                            model_predictions=getattr(theargs, 'model_predictions', None),
                                            disease=getattr(theargs, 'disease', None),
                                            hierarchy=getattr(theargs, 'hierarchy', None),
                                            parent_network=getattr(theargs, 'parent_network', None),
                                            ndexserver=getattr(theargs, 'ndexserver', None),
                                            ndexuser=getattr(theargs, 'ndexuser', None),
                                            ndexpassword=getattr(theargs, 'ndexpassword', None),
                                            visibility=getattr(theargs, 'visibility', False),
                                            gpu=use_gpu,
                                            slurm_partition=slurm_partition,
                                            slurm_account=slurm_account,
                                            input_data_dict=theargs.__dict__
                                            )
        else:
            runner = CellmapsvnnRunner(outdir=theargs.outdir,
                                       command=cmd,
                                       inputdir=theargs.inputdir,
                                       exitcode=theargs.exitcode,
                                       skip_logging=theargs.skip_logging,
                                       input_data_dict=theargs.__dict__)

        return runner.run()
    except Exception as e:
        logger.exception('Caught exception: ' + str(e))
        return 2
    finally:
        logging.shutdown()


def set_arguments_from_config_and_defaults(theargs, config):
    """Sets default values for arguments if not already set."""
    defaults = {
        'gene_attribute_name': vnnconstants.GENE_SET_COLUMN_NAME,
        'batchsize': vnnconstants.DEFAULT_BATCHSIZE,
        'zscore_method': vnnconstants.DEFAULT_ZSCORE_METHOD,
        'epoch': VNNTrain.DEFAULT_EPOCH,
        'lr': VNNTrain.DEFAULT_LR,
        'wd': VNNTrain.DEFAULT_WD,
        'alpha': VNNTrain.DEFAULT_ALPHA,
        'genotype_hiddens': vnnconstants.DEFAULT_GENOTYPE_HIDDENS,
        'patience': VNNTrain.DEFAULT_PATIENCE,
        'delta': VNNTrain.DEFAULT_DELTA,
        'min_dropout_layer': VNNTrain.DEFAULT_MIN_DROPOUT_LAYER,
        'dropout_fraction': VNNTrain.DEFAULT_DROPOUT_FRACTION,
        'optimize': VNNTrain.DEFAULT_OPTIMIZE,
        'n_trials': VNNTrain.DEFAULT_N_TRIALS,
        'cuda': vnnconstants.DEFAULT_CUDA,
        'cpu_count': VNNPredict.DEFAULT_CPU_COUNT,
        'drug_count': VNNPredict.DEFAULT_DRUG_COUNT
    }

    for key in vars(theargs):
        if getattr(theargs, key, None) is None:
            if key in config:
                setattr(theargs, key, config[key])
            elif key in defaults:
                setattr(theargs, key, defaults[key])


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(sys.argv))
