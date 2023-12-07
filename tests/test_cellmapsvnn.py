#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `cellmaps_vnn` package."""
import os
import tempfile
import shutil

import unittest
from unittest.mock import MagicMock

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
            cmd = MagicMock()
            myobj = CellmapsvnnRunner(outdir=os.path.join(temp_dir, 'foo'),
                                      command=cmd,
                                      skip_logging=True,
                                      exitcode=4)
            self.assertEqual(4, myobj.run())
        finally:
            shutil.rmtree(temp_dir)
