import unittest
from unittest.mock import Mock, patch

import networkx as nx
import torch

from cellmaps_vnn.data_wrapper import TrainingDataWrapper
from cellmaps_vnn.vnn import VNN


class TestVNN(unittest.TestCase):

    def setUp(self):
        self.mock_data_wrapper = Mock(spec=TrainingDataWrapper)
        self.mock_data_wrapper.root = 'term1'
        self.mock_data_wrapper.num_hiddens_genotype = 10
        self.mock_data_wrapper.gene_id_mapping = {'gene1': 0, 'gene2': 1}
        self.mock_data_wrapper.term_direct_gene_map = {'term1': {'gene1'}, 'term2': {'gene2'}}
        self.mock_data_wrapper.term_size_map = {'term1': 1, 'term2': 1}
        self.mock_data_wrapper.min_dropout_layer = 1
        self.mock_data_wrapper.dropout_fraction = 0.5
        self.mock_data_wrapper.cell_features = torch.randn(100, 2, 10)
        graph = nx.DiGraph()
        graph.add_edge('term1', 'term2')
        self.mock_data_wrapper.digraph = graph

    def test_cal_term_dim(self):
        vnn = VNN(self.mock_data_wrapper)
        expected_term_dim_map = {'term1': 10, 'term2': 10}
        self.assertEqual(vnn.term_dim_map, expected_term_dim_map)

    def test_construct_direct_gene_layer(self):
        vnn = VNN(self.mock_data_wrapper)
        for gene in self.mock_data_wrapper.gene_id_mapping:
            self.assertIn(gene + '_feature_layer', vnn._modules)
            self.assertIn(gene + '_batchnorm_layer', vnn._modules)

    def test_forward(self):
        vnn = VNN(self.mock_data_wrapper)
        input_tensor = torch.randn(100, 2, 10)
        aux_out_map, hidden_embeddings_map = vnn.forward(input_tensor)

        self.assertIsInstance(aux_out_map, dict)
        self.assertIsInstance(hidden_embeddings_map, dict)
        self.assertIn('term1', hidden_embeddings_map)
        self.assertIn('term2', hidden_embeddings_map)

    def test_construct_nn_graph(self):
        vnn = VNN(self.mock_data_wrapper)
        self.assertIsNotNone(vnn.term_layer_list)
        self.assertIsNotNone(vnn.term_neighbor_map)

    def test_term_layer_addition(self):
        vnn = VNN(self.mock_data_wrapper)
        for term in self.mock_data_wrapper.digraph.nodes():
            self.assertIn(term + '_linear_layer', vnn._modules)

    def test_dropout_layer_addition(self):
        vnn = VNN(self.mock_data_wrapper)
        for i, layer in enumerate(vnn.term_layer_list):
            for term in layer:
                if i >= vnn.min_dropout_layer:
                    self.assertIn(term + '_dropout_layer', vnn._modules)

    def test_batchnorm_layer_addition(self):
        vnn = VNN(self.mock_data_wrapper)
        for term in self.mock_data_wrapper.digraph.nodes():
            self.assertIn(term + '_batchnorm_layer', vnn._modules)

    def test_aux_layer_addition(self):
        vnn = VNN(self.mock_data_wrapper)
        for term in self.mock_data_wrapper.digraph.nodes():
            self.assertIn(term + '_aux_linear_layer1', vnn._modules)
            self.assertIn(term + '_aux_linear_layer2', vnn._modules)

    def test_final_layer_addition(self):
        vnn = VNN(self.mock_data_wrapper)
        self.assertIn('final_aux_linear_layer', vnn._modules)
        self.assertIn('final_linear_layer_output', vnn._modules)


if __name__ == '__main__':
    unittest.main()
