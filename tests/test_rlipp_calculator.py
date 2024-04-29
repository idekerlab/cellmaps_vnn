import os
import shutil
import tempfile
import unittest
import copy
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from scipy import stats
from ndex2.cx2 import RawCX2NetworkFactory
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

import cellmaps_vnn
from cellmaps_vnn.rlipp_calculator import RLIPPCalculator


class TestRLIPPCalculator(unittest.TestCase):
    def setUp(self):
        self.temp_out_dir = tempfile.mkdtemp()
        self.temp_hidden_dir = tempfile.mkdtemp()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_dir = os.path.dirname(self.current_dir)
        self.examples_dir = os.path.join(self.project_dir, 'examples')
        hierarchy_file = os.path.join(self.examples_dir, "hierarchy.cx2")
        factory = RawCX2NetworkFactory()
        hierarchy = factory.get_cx2network(hierarchy_file)
        self.calc = RLIPPCalculator(hierarchy, os.path.join(self.examples_dir, "test_data.txt"),
                                    os.path.join(self.examples_dir, "predict.txt"),
                                    os.path.join(self.examples_dir, "gene2ind.txt"),
                                    os.path.join(self.examples_dir, "cell2ind.txt"),
                                    self.temp_hidden_dir, os.path.join(self.temp_out_dir, 'rlipp.out'),
                                    os.path.join(self.temp_out_dir, 'gene_rho.out'), 1, 4, 0)

    def tearDown(self):
        """Tear down test fixtures, if any."""
        shutil.rmtree(self.temp_hidden_dir)
        shutil.rmtree(self.temp_out_dir)

    def test_create_drug_pos_map(self):
        test_data = {
            'D': ['drug1', 'drug2', 'drug1', 'drug3', 'drug2'],
            'some_other_column': ["A", "B", "C", "D", "E"]
        }
        test_df = pd.DataFrame(test_data)
        calc_copy = copy.copy(self.calc)
        calc_copy.test_df = test_df
        expected_output = {
            'drug1': [0, 2],  # positions of 'drug1'
            'drug2': [1, 4],  # positions of 'drug2'
            'drug3': [3]  # position of 'drug3'
        }
        result = calc_copy.create_drug_pos_map()
        self.assertEqual(result, expected_output)

    def test_create_drug_corr_map_sorted(self):
        test_data = {
            'AUC': [0.9, 0.85, 0.88, 0.95, 0.92]
        }
        test_df = pd.DataFrame(test_data)
        predicted_vals = np.array([0.91, 0.86, 0.89, 0.96, 0.93])
        calc_copy = copy.copy(self.calc)
        calc_copy.test_df = test_df
        calc_copy.predicted_vals = predicted_vals
        drug_pos_map = {
            'drug1': [0, 2],
            'drug2': [1, 4],
            'drug3': []
        }
        expected_output = {
            'drug2': stats.spearmanr([0.85, 0.92], [0.86, 0.93])[0],
            'drug1': stats.spearmanr([0.9, 0.88], [0.91, 0.89])[0],
            'drug3': 0.0
        }
        result = calc_copy.create_drug_corr_map_sorted(drug_pos_map)
        self.assertEqual(result, expected_output)

    def test_load_feature(self):
        element = 'test_element'
        feature_data = """5.7723e-01 5.7793e-01 -5.8209e-01 -5.7857e-01
        2.9274e-01 2.9355e-01 -2.9647e-01 -2.9258e-01
        2.2054e-01 2.2237e-01 -2.2868e-01 -2.2330e-01
        1.0020e-01 9.5688e-02 -9.6748e-02 -9.6321e-02
        -2.1583e-01 -2.1655e-01 2.1733e-01 2.2022e-01
        -1.8145e-01 -1.7938e-01 1.8078e-01 1.8284e-01
        4.7538e-01 4.7915e-01 -4.8235e-01 -4.7642e-01
        -2.7033e-02 -2.5254e-02 2.1806e-02 2.5409e-02"""
        feature_file_path = os.path.join(self.temp_hidden_dir, f'{element}.hidden')
        with open(feature_file_path, 'w') as f:
            f.write(feature_data)

        size = 4
        expected_output = np.array([
            [5.7723e-01, 5.7793e-01, -5.8209e-01, -5.7857e-01],
            [2.9274e-01, 2.9355e-01, -2.9647e-01, -2.9258e-01],
            [2.2054e-01, 2.2237e-01, -2.2868e-01, -2.2330e-01],
            [1.0020e-01, 9.5688e-02, -9.6748e-02, -9.6321e-02],
            [-2.1583e-01, -2.1655e-01, 2.1733e-01, 2.2022e-01],
            [-1.8145e-01, -1.7938e-01, 1.8078e-01, 1.8284e-01],
            [4.7538e-01, 4.7915e-01, -4.8235e-01, -4.7642e-01],
            [-2.7033e-02, -2.5254e-02, 2.1806e-02, 2.5409e-02]
        ])
        loaded_features = self.calc.load_feature(element, size)
        np.testing.assert_array_almost_equal(loaded_features, expected_output)

    def test_create_child_feature_map(self):
        calc_copy = copy.copy(self.calc)
        mock_hierarchy = MagicMock()
        mock_edges = {
            1: {'s': 'term1', 't': 'term3'},
            2: {'s': 'term1', 't': 'term4'},
            3: {'s': 'term2', 't': 'term5'}
        }
        mock_hierarchy.get_edges.return_value = mock_edges
        calc_copy._hierarchy = mock_hierarchy
        feature_map = {
            'term3': 'features_of_term3',
            'term4': 'features_of_term4',
            'term5': 'features_of_term5'
        }
        term = 'term1'
        expected_output = ['term1', 'features_of_term3', 'features_of_term4']

        result = calc_copy.create_child_feature_map(feature_map, term)

        self.assertEqual(result, expected_output)

    def test_get_child_features(self):
        term_child_features = [
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            np.array([0.5, 0.4, 0.3, 0.2, 0.1]),
            np.array([1.0, 0.9, 0.8, 0.7, 0.6])
        ]
        position_map = [1, 3, 4]
        expected_matrix = np.column_stack([
            np.array([0.2, 0.4, 0.5]),
            np.array([0.4, 0.2, 0.1]),
            np.array([0.9, 0.7, 0.6])
        ])
        result_matrix = RLIPPCalculator.get_child_features(term_child_features, position_map)
        np.testing.assert_array_almost_equal(result_matrix, expected_matrix)

    def test_exec_lm(self):
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([2, 3, 4, 5, 6])
        self.calc.num_hiddens_genotype = 2
        correlation, p_value = self.calc.exec_lm(X, y)
        self.assertAlmostEqual(correlation, 1.0, places=5)
        self.assertLess(p_value, 0.05)

    def test_calc_term_rlipp(self):
        calc_copy = copy.copy(self.calc)
        calc_copy.predicted_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        calc_copy.exec_lm = MagicMock(side_effect=[
            (0.9, 0.01),
            (0.8, 0.02)
        ])
        calc_copy.get_child_features = MagicMock(return_value=np.array([0.3, 0.4, 0.5]))
        term_features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        term_child_features = [np.array([1.5, 2.5, 3.5, 4.5, 5.5])]
        position_map = [1, 3, 4]
        term = "GeneA"
        drug = "DrugX"
        expected_output = "GeneA\t9.000e-01\t1.000e-02\t8.000e-01\t2.000e-02\t1.125e+00\n"
        result = calc_copy.calc_term_rlipp(term_features, term_child_features, position_map, term, drug)
        self.assertEqual(result, expected_output)

    def test_calc_gene_rho(self):
        calc_copy = copy.copy(self.calc)
        calc_copy.predicted_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        gene_features = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        position_map = [0, 2, 4]
        gene = "GeneX"
        drug = "DrugY"
        pred = np.take(calc_copy.predicted_vals, position_map)
        gene_embeddings = np.take(gene_features, position_map)
        rho, p_val = stats.spearmanr(pred, gene_embeddings)
        expected_output = '{}\t{:.3e}\t{:.3e}\n'.format(gene, rho, p_val)
        result = calc_copy.calc_gene_rho(gene_features, position_map, gene, drug)
        self.assertEqual(result, expected_output)


if __name__ == '__main__':
    unittest.main()
