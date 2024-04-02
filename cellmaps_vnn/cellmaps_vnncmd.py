#! /usr/bin/env python

import argparse
import sys
import logging
import logging.config
from cellmaps_utils import logutils
from cellmaps_utils import constants
import cellmaps_vnn
from cellmaps_vnn.annotate import VNNAnnotate
from cellmaps_vnn.predict import VNNPredict
from cellmaps_vnn.runner import CellmapsvnnRunner
from cellmaps_vnn.train import VNNTrain

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

    try:
        logutils.setup_cmd_logging(theargs)

        if theargs.command == VNNTrain.COMMAND:
            cmd = VNNTrain(theargs)
        elif theargs.command == VNNPredict.COMMAND:
            cmd = VNNPredict(theargs)
        elif theargs.command == VNNAnnotate.COMMAND:
            cmd = VNNAnnotate(theargs)
            theargs.inputdir = theargs.model_predictions
        else:
            raise Exception('Invalid command: ' + str(theargs.command))

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


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(sys.argv))
