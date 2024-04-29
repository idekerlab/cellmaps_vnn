import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

from cellmaps_vnn.annotate import VNNAnnotate


class TestVNNAnnotate(unittest.TestCase):
    """Tests for the VNNAnnotate class in the cellmaps_vnn package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.mock_args = MagicMock()
        self.mock_args.outdir = '/fake/output/directory'
        self.mock_args.model_predictions = ['/fake/model_predictions']
        self.mock_args.hierarchy = '/fake/hierarchy.cx2'
        self.mock_args.disease = None
        self.mock_args.upload_to_ndex = False

    def tearDown(self):
        """Tear down test fixtures, if any."""

    @patch('pandas.read_csv')
    def test_get_scores_for_disease(self, mock_read_csv):
        data = {'Term': ['Term1', 'Term2'], 'P_rho': [0.5, 0.7], 'Disease': ['Cancer', 'Other']}
        test_df = pd.DataFrame(data)
        mock_read_csv.return_value = test_df
        annotator = VNNAnnotate(self.mock_args)
        result = annotator._get_scores_for_disease('Cancer')
        self.assertEqual(result, {'Term1': 0.5})


if __name__ == '__main__':
    unittest.main()
