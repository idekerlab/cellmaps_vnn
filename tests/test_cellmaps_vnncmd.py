#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `cellmaps_vnn` package."""

import argparse
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

    def _create_rocrate(self, directory, files=None):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, 'ro-crate-metadata.json'), 'w') as f:
            f.write('{}')
        if files is None:
            return
        for relpath, content in files.items():
            filepath = os.path.join(directory, relpath)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(content)

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

    def test_parse_arguments_mode_train(self):
        """Tests parse arguments in mode interface for train"""
        res = cellmaps_vnncmd._parse_arguments(
            'hi',
            ['outdir', '--mode', 'train', '--input_crate', 'crate_dir']
        )

        self.assertEqual('train', res.command)
        self.assertEqual('train', res.mode)
        self.assertEqual('crate_dir', res.inputdir)
        self.assertEqual('crate_dir', res.input_crate)
        self.assertEqual('mode', res.command_source)

    def test_parse_arguments_mode_test(self):
        """Tests parse arguments in mode interface for test"""
        res = cellmaps_vnncmd._parse_arguments(
            'hi',
            ['outdir', '--mode', 'test', '--input_crate', 'input_crate', '--model', 'model_crate']
        )

        self.assertEqual('predict', res.command)
        self.assertEqual('test', res.mode)
        self.assertEqual(['model_crate', 'input_crate'], res.inputdir)
        self.assertEqual('mode', res.command_source)

    def test_parse_arguments_mode_missing_model(self):
        """Ensures missing model raises SystemExit during parsing"""
        with self.assertRaises(SystemExit):
            cellmaps_vnncmd._parse_arguments(
                'hi',
                ['outdir', '--mode', 'predict', '--input_crate', 'input_crate']
            )

    def test_prepare_mode_inputs_predict_sets_defaults(self):
        """Ensures predict mode sets config file and absolute paths"""
        temp_dir = tempfile.mkdtemp()
        try:
            input_crate = os.path.join(temp_dir, 'input')
            model_crate = os.path.join(temp_dir, 'model')
            self._create_rocrate(input_crate)
            self._create_rocrate(model_crate, {
                cellmaps_vnncmd.vnnconstants.MODEL_FILENAME: 'model',
                cellmaps_vnncmd.vnnconstants.CONFIG_FILENAME: 'conf'
            })
            args = argparse.Namespace(
                command=cellmaps_vnncmd.VNNPredict.COMMAND,
                mode='test',
                command_source='mode',
                input_crate=input_crate,
                model_crate=model_crate,
                inputdir=None,
                config_file=None
            )
            cellmaps_vnncmd._prepare_mode_inputs(args)
            self.assertEqual(
                [os.path.abspath(model_crate), os.path.abspath(input_crate)],
                args.inputdir
            )
            self.assertEqual(
                os.path.join(os.path.abspath(model_crate), cellmaps_vnncmd.vnnconstants.CONFIG_FILENAME),
                args.config_file
            )
        finally:
            shutil.rmtree(temp_dir)

    def test_prepare_mode_inputs_requires_rocrate_manifest(self):
        """Ensures missing ro-crate manifest raises error"""
        temp_dir = tempfile.mkdtemp()
        try:
            args = argparse.Namespace(
                command=cellmaps_vnncmd.VNNTrain.COMMAND,
                mode='train',
                command_source='mode',
                input_crate=os.path.join(temp_dir, 'input'),
                inputdir=None,
                config_file=None
            )
            os.makedirs(args.input_crate)
            with self.assertRaises(cellmaps_vnncmd.CellmapsvnnError):
                cellmaps_vnncmd._prepare_mode_inputs(args)
        finally:
            shutil.rmtree(temp_dir)

    def test_normalize_optimize_flag_falls_back(self):
        """Ensures optimizetrain with no ranges disables optimization"""
        args = argparse.Namespace(
            command=cellmaps_vnncmd.VNNTrain.COMMAND,
            mode='optimizetrain',
            command_source='mode',
            optimize=1,
            batchsize=64,
            lr=0.001,
            wd=0.001,
            alpha=0.3,
            genotype_hiddens=4,
            patience=30,
            delta=0.001,
            min_dropout_layer=2,
            dropout_fraction=0.3
        )
        config = {'optimize': 1}
        cellmaps_vnncmd._normalize_optimize_flag(args, config)
        self.assertEqual(0, args.optimize)
        self.assertEqual(0, config['optimize'])

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
