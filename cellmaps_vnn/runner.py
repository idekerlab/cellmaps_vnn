#! /usr/bin/env python

import os
import time
import logging
from cellmaps_utils import logutils, constants
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
