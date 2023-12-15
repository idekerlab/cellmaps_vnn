import os.path
import sys

import ndex2
import networkx as nx
import numpy as np
import pandas as pd
import random as rd
import torch
from ndex2.cx2 import RawCX2NetworkFactory, CX2NetworkXFactory

import cellmaps_vnn.util as util
from cellmaps_vnn.exceptions import CellmapsvnnError


class TrainingDataWrapper:

    def __init__(self, theargs):

        self.root = None
        self.digraph = None
        self.num_hiddens_genotype = theargs.genotype_hiddens
        self.lr = theargs.lr
        self.wd = theargs.wd
        self.alpha = theargs.alpha
        self.epochs = theargs.epoch
        self.batchsize = theargs.batchsize
        self.modeldir = self._setup_model_dir(theargs.modeldir)
        self.cuda = theargs.cuda
        self.zscore_method = theargs.zscore_method
        self.std = theargs.std
        self.patience = theargs.patience
        self.delta = theargs.delta
        self.min_dropout_layer = theargs.min_dropout_layer
        self.dropout_fraction = theargs.dropout_fraction

        self._hierarchy = theargs.hierarchy
        self._hierarchy_parent = theargs.hierarchy_parent  # not sure if needed
        self._training_data = theargs.training_data
        self.cell_id_mapping = util.load_mapping(theargs.cell2id, 'cell lines')
        self.gene_id_mapping = util.load_mapping(theargs.gene2id, 'genes')

        self.cell_features = util.load_cell_features(theargs.mutations, theargs.cn_deletions, theargs.cn_amplifications)
        self.train_feature, self.train_label, self.val_feature, self.val_label = self._prepare_train_data()
        self._load_graph(self._hierarchy)

    @staticmethod
    def _setup_model_dir(model_dir):
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        return model_dir

    def _prepare_train_data(self):
        train_features, train_labels, val_features, val_labels = self._load_train_data()
        return (torch.Tensor(train_features), torch.FloatTensor(train_labels), torch.Tensor(val_features),
                torch.FloatTensor(val_labels))

    def _load_train_data(self):
        all_df = pd.read_csv(self._training_data, sep='\t', header=None,
                             names=['cell_line', 'smiles', 'auc', 'dataset'])
        train_df, val_df = self._split_train_val_data(all_df)
        std_df = util.calc_std_vals(train_df, self.zscore_method)
        std_df.to_csv(self.std, sep='\t', header=False, index=False)
        train_df = util.standardize_data(train_df, std_df)
        val_df = util.standardize_data(val_df, std_df)
        train_features, train_labels = self._extract_features_labels(train_df)
        val_features, val_labels = self._extract_features_labels(val_df)
        return train_features, train_labels, val_features, val_labels

    def _split_train_val_data(self, all_df):
        train_cell_lines = list(set(all_df['cell_line']))
        val_cell_lines = self._select_validation_cell_lines(train_cell_lines)
        val_df = all_df.query('cell_line in @val_cell_lines').reset_index(drop=True)
        train_df = all_df.query('cell_line in @train_cell_lines').reset_index(drop=True)
        return train_df, val_df

    @staticmethod
    def _select_validation_cell_lines(train_cell_lines):
        val_size = int(len(train_cell_lines) / 5)
        val_cell_lines = []
        for _ in range(val_size):
            r = rd.randint(0, len(train_cell_lines) - 1)
            val_cell_lines.append(train_cell_lines.pop(r))
        return val_cell_lines

    def _extract_features_labels(self, df):
        features = []
        labels = []
        for row in df.values:
            features.append([self.cell_id_mapping[row[0]]])
            labels.append([float(row[2])])
        return features, labels

    def _load_graph(self, file_name):

        try:
            digraph, cx2network = self._create_digraph(file_name)
            roots = [n for n in digraph.nodes if digraph.in_degree(n) == 0]
            ugraph = digraph.to_undirected()
            connected_sub_graph_list = list(nx.connected_components(ugraph))

            if len(roots) != 1 or len(connected_sub_graph_list) != 1:
                raise CellmapsvnnError("Graph must have exactly one root and be fully connected")

            self.digraph = self._convert_graph_to_string_nodes(digraph)
            self.root = str(roots[0])
            self._generate_term_maps(cx2network)

        except Exception as e:
            raise CellmapsvnnError(f"Error loading graph: {e}")

    @staticmethod
    def _create_digraph(file_name):
        cx2factory = RawCX2NetworkFactory()
        nxfactory = CX2NetworkXFactory()
        cx2network = cx2factory.get_cx2network(file_name)
        digraph = nxfactory.get_graph(cx2network, nx.DiGraph())
        return digraph, cx2network

    def _generate_term_maps(self, cx2_network):
        term_direct_gene_map = self._get_direct_genes(cx2_network)
        term_size_map = {}

        for term in self.digraph.nodes():
            term_gene_set = term_direct_gene_map.get(term, set())
            descendants = nx.descendants(self.digraph, term)
            for child in descendants:
                if child in term_direct_gene_map:
                    term_gene_set = term_gene_set | term_direct_gene_map[child]

            if len(term_gene_set) == 0:
                raise CellmapsvnnError(f'There is an empty term, please delete term: {term}')
            else:
                term_size_map[term] = len(term_gene_set)

        self.term_size_map = term_size_map
        self.term_direct_gene_map = term_direct_gene_map

    def _get_direct_genes(self, cx2_network):
        term_direct_gene_map = {}
        child_genes_map = {}

        for edge_id, edge_data in cx2_network.get_edges().items():
            parent_node_id = edge_data['s']
            child_node_id = edge_data['t']

            if parent_node_id not in child_genes_map:
                child_genes_map[parent_node_id] = set()

            child_node_genes = self._get_genes_of_node(cx2_network, child_node_id)
            child_genes_map[parent_node_id].update(child_node_genes)

        for node_id, node_data in cx2_network.get_nodes().items():
            node_id_str = str(node_id)
            all_genes = self._get_genes_of_node(cx2_network, node_id)
            direct_genes = all_genes - child_genes_map.get(node_id, set())
            if len(direct_genes) > 0:
                term_direct_gene_map[node_id_str] = direct_genes

        return term_direct_gene_map

    def _get_genes_of_node(self, cx2_network, node_id):
        """
        Retrieves genes associated with a specific node.
        """
        genes = set()
        node_data = cx2_network.get_node(node_id)

        if node_data and 'CD_MemberList' in node_data[ndex2.constants.ASPECT_VALUES]:
            for gene_identifier in node_data[ndex2.constants.ASPECT_VALUES]['CD_MemberList'].split():
                if gene_identifier in self.gene_id_mapping:
                    genes.add(self.gene_id_mapping[gene_identifier])

        return genes

    @staticmethod
    def _convert_graph_to_string_nodes(original_graph):
        """
        Converts a graph with integer nodes to a graph with string nodes.

        Parameters:
        original_graph (nx.Graph): The original graph with integer nodes.

        Returns:
        nx.Graph: A new graph with the same structure but with string nodes.
        """
        new_graph = type(original_graph)()

        for node in original_graph.nodes():
            new_graph.add_node(str(node))

        for edge in original_graph.edges():
            new_graph.add_edge(str(edge[0]), str(edge[1]))

        return new_graph
