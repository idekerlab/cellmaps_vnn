#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `cellmaps_vnn` package."""
import os
import tempfile
import shutil

import unittest
from unittest.mock import MagicMock

from cellmaps_utils.provenance import ProvenanceUtil

from cellmaps_vnn.runner import CellmapsvnnRunner


class TestCellmapsvnnrunner(unittest.TestCase):
    """Tests for `cellmaps_vnn` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_constructor(self):
        """Tests constructor"""
        myobj = CellmapsvnnRunner(outdir='foo', skip_logging=True,
                                  exitcode=0)

        self.assertIsNotNone(myobj)

    def test_run(self):
        """ Tests run()"""
        temp_dir = tempfile.mkdtemp()
        try:
            hier_dir = os.path.join(temp_dir, '4.hierarchy')
            os.makedirs(hier_dir, mode=0o755)
            prov = ProvenanceUtil()
            prov.register_rocrate(hier_dir, name='hierarchy1',
                                  organization_name='hierarchy org',
                                  project_name='hierarchy project')
            cmd = MagicMock()
            cmd.COMMAND = 'train'
            myobj = CellmapsvnnRunner(outdir=os.path.join(temp_dir, 'foo'),
                                      inputdir=hier_dir,
                                      command=cmd,
                                      skip_logging=True,
                                      exitcode=4)
            self.assertEqual(4, myobj.run())
        finally:
            shutil.rmtree(temp_dir)
