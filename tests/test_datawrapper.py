import os
import shutil
import tempfile
import unittest
import types
from unittest import skip
from unittest.mock import MagicMock, patch

import ndex2.cx2
import networkx as nx
import pandas as pd
import torch

from cellmaps_vnn.data_wrapper import TrainingDataWrapper
import cellmaps_vnn.util as util


class MockArgs:
    def __init__(self, examples_dir, temp_dir):
        self.genotype_hiddens = 128
        self.lr = 0.001
        self.wd = 0.01
        self.alpha = 0.5
        self.epoch = 10
        self.batchsize = 32
        self.outdir = temp_dir
        self.cuda = False
        self.zscore_method = 'standard'
        self.stdfile = os.path.join(temp_dir, 'std.txt')
        self.modelfile = os.path.join(temp_dir, 'model_final.pt')
        self.patience = 5
        self.delta = 0.1
        self.min_dropout_layer = 1
        self.dropout_fraction = 0.3
        self.inputdir = examples_dir
        self.hierarchy = os.path.join(examples_dir, 'hierarchy.cx2')
        self.hierarchy_parent = os.path.join(examples_dir, 'hierarchy_parent.cx2')
        self.training_data = os.path.join(examples_dir, 'training_data.txt')
        self.cell2id = os.path.join(examples_dir, 'cell2ind.txt')
        self.gene2id = os.path.join(examples_dir, 'gene2ind.txt')
        self.mutations = os.path.join(examples_dir, 'cell2mutation.txt')
        self.cn_deletions = os.path.join(examples_dir, 'cell2cndeletion.txt')
        self.cn_amplifications = os.path.join(examples_dir, 'cell2cnamplification.txt')
        self.gene_attribute_name = 'CD_MemberList'


class TestTrainingDataWrapper(unittest.TestCase):

    def setUp(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_dir = os.path.dirname(self.current_dir)
        self.examples_dir = os.path.join(self.project_dir, 'examples')
        self.temp_dir = tempfile.mkdtemp()
        self.theargs = MockArgs(self.examples_dir, self.temp_dir)

    def tearDown(self):
        """Tear down test fixtures, if any."""
        shutil.rmtree(self.temp_dir)
        patch.stopall()

    @patch('cellmaps_vnn.data_wrapper.TrainingDataWrapper._prepare_train_data',
           return_value=(['mock_train_feature'], ['mock_train_label'], ['mock_val_feature'], ['mock_val_label']))
    @patch('cellmaps_vnn.data_wrapper.TrainingDataWrapper._load_graph')
    def test_init_with_mocks(self, mock_load_graph, mock_prepare_train_data):
        mock_load_mapping = MagicMock()
        mock_load_numpy_data = MagicMock()
        mock_load_mapping.return_value = "mocked_mapping_value"
        mock_load_numpy_data.return_value = "mocked_numpy_data"
        util.load_mapping = mock_load_mapping
        util.load_numpy_data = mock_load_numpy_data
        wrapper = TrainingDataWrapper(self.theargs)
        self.assertEqual(wrapper._hierarchy, self.theargs.hierarchy)
        self.assertEqual(wrapper.num_hiddens_genotype, self.theargs.genotype_hiddens)
        self.assertEqual(wrapper.lr, self.theargs.lr)
        mock_prepare_train_data.assert_called_once()
        mock_load_graph.assert_called_once()

    def test_init(self):
        wrapper = TrainingDataWrapper(self.theargs)
        self.assertEqual(wrapper._hierarchy, self.theargs.hierarchy)
        self.assertEqual(wrapper.num_hiddens_genotype, self.theargs.genotype_hiddens)
        self.assertEqual(wrapper.lr, self.theargs.lr)
        self.assertIsNotNone(wrapper.train_feature)

    def test_prepare_train_data(self):
        mock_train_features = [[1, 2, 3], [4, 5, 6]]
        mock_train_labels = [0, 1]
        mock_val_features = [[7, 8, 9], [10, 11, 12]]
        mock_val_labels = [1, 0]
        wrapper = MagicMock()
        wrapper._prepare_train_data = types.MethodType(TrainingDataWrapper._prepare_train_data, wrapper)
        wrapper._load_train_data.return_value = (mock_train_features, mock_train_labels, mock_val_features,
                                                 mock_val_labels)
        train_features, train_labels, val_features, val_labels = wrapper._prepare_train_data()
        self.assertTrue(isinstance(train_features, torch.Tensor))
        self.assertTrue(isinstance(train_labels, torch.FloatTensor))
        self.assertTrue(isinstance(val_features, torch.Tensor))
        self.assertTrue(isinstance(val_labels, torch.FloatTensor))

    def test_select_validation_cell_lines(self):
        mock_train_cell_lines = ['CellLine1', 'CellLine2', 'CellLine3', 'CellLine4', 'CellLine5']
        expected_val_size = int(len(mock_train_cell_lines) / 5)
        expected_remaining_train_size = len(mock_train_cell_lines) - expected_val_size
        val_cell_lines = TrainingDataWrapper._select_validation_cell_lines(mock_train_cell_lines)
        self.assertEqual(len(val_cell_lines), expected_val_size)
        for val_cell_line in val_cell_lines:
            self.assertNotIn(val_cell_line, mock_train_cell_lines)
        self.assertEqual(len(mock_train_cell_lines), expected_remaining_train_size)

    def test_create_digraph(self):
        digraph, cx2network = TrainingDataWrapper._create_digraph(self.theargs.hierarchy)
        self.assertTrue(isinstance(digraph, nx.DiGraph))
        self.assertTrue(isinstance(cx2network, ndex2.cx2.CX2Network))


if __name__ == '__main__':
    unittest.main()
