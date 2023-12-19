import unittest
from unittest.mock import patch, mock_open

from cellmaps_vnn.util import *


class TestUtil(unittest.TestCase):

    def test_calc_std_vals_zscore(self):
        df = pd.DataFrame({'dataset': ['A', 'A', 'B', 'B'],
                           'auc': [1.0, 2.0, 3.0, 4.0]})
        expected_result = pd.DataFrame({
            'dataset': ['A', 'B'],
            'center': [1.5, 3.5],
            'scale': [0.707107, 0.707107]
        }, columns=['dataset', 'center', 'scale'])
        result = calc_std_vals(df, 'zscore')
        pd.testing.assert_frame_equal(result, expected_result)

    def test_calc_std_vals_robustz(self):
        df = pd.DataFrame({'dataset': ['A', 'A', 'B', 'B'],
                           'auc': [1.0, 2.0, 3.0, 4.0]})
        expected_result = pd.DataFrame({
            'dataset': ['A', 'B'],
            'center': [1.5, 3.5],
            'scale': [0.5, 0.5]
        }, columns=['dataset', 'center', 'scale'])
        result = calc_std_vals(df, 'robustz')
        pd.testing.assert_frame_equal(result, expected_result)

    def test_standardize_data(self):
        df = pd.DataFrame({
            'dataset': ['X', 'X', 'Y', 'Y'],
            'auc': [1.0, 2.0, 3.0, 4.0],
            'cell_line': ['CL1', 'CL2', 'CL1', 'CL2'],
            'smiles': ['C1', 'C2', 'C1', 'C2']
        })
        std_df = pd.DataFrame({
            'dataset': ['X', 'Y'],
            'center': [1.5, 3.5],
            'scale': [0.5, 0.5]
        })
        expected_result = pd.DataFrame({
            'cell_line': ['CL1', 'CL2', 'CL1', 'CL2'],
            'smiles': ['C1', 'C2', 'C1', 'C2'],
            'z': [-1.0, 1.0, -1.0, 1.0]
        })
        result = standardize_data(df, std_df)
        pd.testing.assert_frame_equal(result, expected_result)

    @patch('os.path.isfile', return_value=True)
    @patch('numpy.genfromtxt')
    def test_load_numpy_data_success(self, mock_genfromtxt, mock_isfile):
        test_file_path = "test.csv"
        test_data = np.array([[1, 2], [3, 4]])
        mock_genfromtxt.return_value = test_data

        result = load_numpy_data(test_file_path)
        np.testing.assert_array_equal(result, test_data)

    @patch('cellmaps_vnn.util.load_numpy_data')
    def test_load_cell_features(self, mock_load_numpy_data):
        test_mutations = np.array([[1, 0], [0, 1]])
        test_cn_deletions = np.array([[2, 3], [3, 4]])
        test_cn_amplifications = np.array([[4, 5], [5, 6]])

        mock_load_numpy_data.side_effect = [test_mutations, test_cn_deletions, test_cn_amplifications]

        expected_result = np.dstack([test_mutations, test_cn_deletions, test_cn_amplifications])

        result = load_cell_features("mutations.csv", "deletions.csv", "amplifications.csv")
        np.testing.assert_array_equal(result, expected_result)

    def test_build_input_vector(self):
        input_data = torch.tensor([[0], [1]])
        cell_features = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

        expected_feature = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float)

        result = build_input_vector(input_data, cell_features)
        torch.testing.assert_allclose(result, expected_feature)

    def test_get_grad_norm(self):
        param = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
        param.grad = torch.tensor([0.1, -0.2, 0.3])

        norm_type = 2
        expected_norm = torch.norm(torch.tensor([0.1, 0.2, 0.3]), norm_type)

        result = get_grad_norm(param, norm_type)
        self.assertAlmostEqual(result.item(), expected_norm.item(), places=5)

    def test_pearson_corr(self):
        x = torch.tensor([1, 2, 3], dtype=torch.float)
        y = torch.tensor([1, 2, 3], dtype=torch.float)

        # Pearson correlation of identical vectors should be 1
        expected_corr = torch.tensor(1.0)

        result = pearson_corr(x, y)
        self.assertAlmostEqual(result.item(), expected_corr.item(), places=5)


if __name__ == '__main__':
    unittest.main()
