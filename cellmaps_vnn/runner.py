#! /usr/bin/env python

import os
import time
import logging
from cellmaps_utils import logutils
from cellmaps_utils.provenance import ProvenanceUtil

import cellmaps_vnn
from cellmaps_vnn.exceptions import CellmapsvnnError

logger = logging.getLogger(__name__)


class CellmapsvnnRunner(object):
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
        if outdir is None:
            raise CellmapsvnnError('outdir is None')

        self._outdir = os.path.abspath(outdir)
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

        :return:
        """
        prov_attrs = self._provenance_utils.get_merged_rocrate_provenance_attrs(self._inputdir,
                                                                                override_name=self._name,
                                                                                override_project_name=
                                                                                self._project_name,
                                                                                override_organization_name=
                                                                                self._organization_name,
                                                                                extra_keywords=[
                                                                                    'VNN',
                                                                                    'Visible Neural Network',
                                                                                    str(self._command)
                                                                                    ])
        if self._name is None:
            self._name = prov_attrs.get_name()

        if self._organization_name is None:
            self._organization_name = prov_attrs.get_organization_name()

        if self._project_name is None:
            self._project_name = prov_attrs.get_project_name()
        self._keywords = prov_attrs.get_keywords()
        self._description = prov_attrs.get_description()

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

            if self._command:
                self._command.run()
            else:
                raise CellmapsvnnError("No command provided to CellmapsvnnRunner")

            # set exit code to value passed in via constructor
            exitcode = self._exitcode if self._exitcode is not None else 0
        finally:
            # write a task finish file
            logutils.write_task_finish_json(outdir=self._outdir,
                                            start_time=self._start_time,
                                            status=exitcode)

        return exitcode
