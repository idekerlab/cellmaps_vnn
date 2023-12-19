import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from cellmaps_vnn.predict import VNNPredict


class TestPredict(unittest.TestCase):
    """Tests for `predict` in `cellmaps_vnn` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.mock_args = MagicMock()
        self.mock_args.modeldir = '/path/to/model'
        self.mock_args.outdir = '/path/to/output'
        self.mock_args.predict_data = '/path/to/predict_data'
        self.mock_args.cell2id = '/path/to/cell2id'
        self.mock_args.mutations = '/path/to/mutations'
        self.mock_args.cn_deletions = '/path/to/cn_deletions'
        self.mock_args.cn_amplifications = '/path/to/cn_amplifications'
        self.mock_args.batchsize = 1000
        self.mock_args.cuda = 0
        self.mock_args.zscore_method = 'auc'
        self.mock_args.std = None

    def tearDown(self):
        """Tear down test fixtures, if any."""

    @patch('cellmaps_vnn.util.load_mapping')
    @patch('cellmaps_vnn.predict.VNNPredict._load_pred_data')
    def test_prepare_predict_data(self, mock_load_pred_data, mock_load_mapping):
        vnn_predict = VNNPredict(self.mock_args)
        mock_load_mapping.return_value = {'cell_line': 'cell_id'}
        mock_load_pred_data.return_value = ([0.5], [1])
        predict_data, cell2id_mapping = vnn_predict._prepare_predict_data(
            self.mock_args.predict_data, self.mock_args.cell2id, self.mock_args.zscore_method, self.mock_args.std)

        self.assertIsInstance(predict_data, tuple)
        self.assertIsInstance(cell2id_mapping, dict)

    @patch('cellmaps_vnn.predict.pd.read_csv')
    @patch('cellmaps_vnn.util.calc_std_vals')
    @patch('cellmaps_vnn.util.standardize_data')
    def test_load_pred_data(self, mock_standardize_data, mock_calc_std_vals, mock_read_csv):
        mock_train_std_df = pd.DataFrame({
            'dataset': ['dataset1'],
            'center': [0.5],
            'scale': [1.0]
        })
        mock_test_df = pd.DataFrame({
            'cell_line': ['cell1', 'cell2'],
            'smiles': ['smile1', 'smile2'],
            'auc': [0.6, 0.7],
            'dataset': ['dataset1', 'dataset1']
        })
        mock_read_csv.side_effect = [mock_train_std_df, mock_test_df]
        mock_calc_std_vals.return_value = mock_test_df.copy()
        mock_standardize_data.return_value = mock_test_df.copy()
        cell2id = {'cell1': 1, 'cell2': 2}

        feature, label = VNNPredict._load_pred_data('test.csv', cell2id, 'auc', 'train_std.csv')

        self.assertEqual(feature, [[1], [2]])
        self.assertEqual(label, [[0.6], [0.7]])

