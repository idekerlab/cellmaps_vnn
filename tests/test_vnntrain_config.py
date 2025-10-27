#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import yaml

from cellmaps_vnn.train import VNNTrain


class TestVNNTrainConfig(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _build_train_stub(self, optimize=0):
        train = object.__new__(VNNTrain)
        train._outdir = self.temp_dir
        train._inputdir = '/input'
        train._config_file = None
        train._gene_attribute_name = 'CD_MemberList'
        train._training_data = '/tmp/training.txt'
        train._gene2id = '/tmp/gene2id.txt'
        train._cell2id = '/tmp/cell2id.txt'
        train._mutations = '/tmp/mutations.txt'
        train._cn_deletions = '/tmp/cn_del.txt'
        train._cn_amplifications = '/tmp/cn_amp.txt'
        train._hierarchy = '/tmp/hierarchy.cx2'
        train._parent_network = '/tmp/parent.cx2'
        train._batchsize = 64
        train._epoch = 50
        train._zscore_method = 'auc'
        train._lr = 0.001
        train._wd = 0.001
        train._alpha = 0.3
        train._genotype_hiddens = 4
        train._patience = 30
        train._delta = 0.001
        train._min_dropout_layer = 2
        train._dropout_fraction = 0.3
        train._optimize = optimize
        train._n_trials = 3
        train._cuda = 0
        train._skip_parent_copy = False
        train._slurm = False
        train._use_gpu = False
        train._slurm_partition = None
        train._slurm_account = None
        train._stdfile = os.path.join(self.temp_dir, 'std.txt')
        return train

    def test_save_final_config_without_source_config(self):
        train = self._build_train_stub()
        config_path = train._save_final_config()
        self.assertTrue(os.path.isfile(config_path))
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        self.assertEqual(64, data['batchsize'])
        self.assertEqual(0.001, data['lr'])
        self.assertIn('optimize', data)

    def test_save_final_config_merges_existing_config(self):
        train = self._build_train_stub()
        source_config = os.path.join(self.temp_dir, 'source.yaml')
        with open(source_config, 'w') as f:
            yaml.safe_dump({'custom': 123, 'lr': 0.5}, f)
        train._config_file = source_config
        config_path = train._save_final_config()
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        self.assertEqual(123, data['custom'])
        self.assertEqual(0.001, data['lr'])

    def test_apply_trial_params_updates_values(self):
        train = self._build_train_stub(optimize=1)
        wrapper = SimpleNamespace(
            lr=0.005,
            batchsize=32,
            num_hiddens_genotype=4,
            patience=20,
            delta=0.01,
            min_dropout_layer=2,
            dropout_fraction=0.2,
            wd=0.002,
            alpha=0.4
        )
        params = {'lr': 0.01, 'genotype_hiddens': 6, 'batchsize': 128}
        train._apply_trial_params(params, wrapper)
        self.assertEqual(0.01, train._lr)
        self.assertEqual(0.01, wrapper.lr)
        self.assertEqual(6, train._genotype_hiddens)
        self.assertEqual(6, wrapper.num_hiddens_genotype)
        self.assertEqual(128, train._batchsize)
        self.assertEqual(128, wrapper.batchsize)
