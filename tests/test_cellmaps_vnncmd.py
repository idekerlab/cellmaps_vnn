#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `cellmaps_vnn` package."""

import os
import tempfile
import shutil

import unittest
from cellmaps_vnn import cellmaps_vnncmd


class TestCellmaps_vnn(unittest.TestCase):
    """Tests for `cellmaps_vnn` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_parse_arguments(self):
        """Tests parse arguments"""
        res = cellmaps_vnncmd._parse_arguments('hi', ['train', 'outdir',
                                                      '--inputdir', 'foodir',
                                                      '--training_data', 'footrain',
                                                      '--gene2id', 'foo', '--cell2id', 'foo',
                                                      '--mutations', 'foo', '--cn_deletions', 'foo',
                                                      '--cn_amplifications', 'foo'])

        self.assertEqual('outdir', res.outdir)
        self.assertEqual(1, res.verbose)
        self.assertEqual(0, res.exitcode)
        self.assertEqual(None, res.logconf)

        someargs = ['-vv', '--logconf', 'hi', '--exitcode', '3', 'train', 'outdir',
                    '--inputdir', 'foodir',
                    '--training_data', 'footrain',
                    '--gene2id', 'foo', '--cell2id', 'foo',
                    '--mutations', 'foo', '--cn_deletions', 'foo',
                    '--cn_amplifications', 'foo']
        res = cellmaps_vnncmd._parse_arguments('hi', someargs)

        self.assertEqual('outdir', res.outdir)
        self.assertEqual(3, res.verbose)
        self.assertEqual('hi', res.logconf)
        self.assertEqual(3, res.exitcode)

    def test_main(self):
        """Tests main function"""

        temp_dir = tempfile.mkdtemp()
        # try where loading config is successful
        try:
            outdir = os.path.join(temp_dir, 'out')
            res = cellmaps_vnncmd.main(['myprog.py', '--skip_logging', 'train', outdir,
                                        '--inputdir', 'foodir',
                                        '--training_data', 'footrain',
                                        '--gene2id', 'foo', '--cell2id', 'foo',
                                        '--mutations', 'foo', '--cn_deletions', 'foo',
                                        '--cn_amplifications', 'foo'])
            self.assertEqual(res, 2)
        finally:
            shutil.rmtree(temp_dir)
