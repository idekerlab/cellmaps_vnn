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


    @patch('cellmaps_vnn.annotate.pd.read_csv')
    def test_get_scores_for_disease(self, mock_read_csv):
        """Test filtering disease-specific scores from RLIPP output."""
        data = {
            'Term': ['Term1', 'Term1'],
            'P_rho': [0.5, 0.7],
            'P_pval': [0.01, 0.02],
            'C_rho': [0.6, 0.8],
            'C_pval': [0.03, 0.04],
            'RLIPP': [1.0, 1.5],
            'Disease': ['Cancer', 'Other']
        }
        mock_read_csv.return_value = pd.DataFrame(data)

        annotator = VNNAnnotate(self.mock_args.outdir, self.mock_args.model_predictions,
                                hierarchy=self.mock_args.hierarchy, disease='Cancer')
        annotator._get_rlipp_out_dest_file = MagicMock(return_value='/fake/path/rlipp.txt')

        result = annotator._get_scores_for_disease('Cancer')

        expected = {
            'Term1': [0.5, 0.01, 0.6, 0.03, 1.0]
        }
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
