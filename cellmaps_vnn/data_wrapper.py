import sys
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import numpy as np
import pandas as pd
import random as rd
import torch
from ndex2.cx2 import RawCX2NetworkFactory, CX2NetworkXFactory

import cellmaps_vnn.util as util


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
        self.modeldir = theargs.modeldir
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
        self.mutations = np.genfromtxt(theargs.mutations, delimiter=',')
        self.cn_deletions = np.genfromtxt(theargs.cn_deletions, delimiter=',')
        self.cn_amplifications = np.genfromtxt(theargs.cn_amplifications, delimiter=',')

        self.cell_features = np.dstack([self.mutations, self.cn_deletions, self.cn_amplifications])
        self.train_feature, self.train_label, self.val_feature, self.val_label = self._prepare_train_data()
        self._load_graph(self._hierarchy)

    def _prepare_train_data(self):
        train_features, val_features, train_labels, val_labels = self._load_train_data()
        return (torch.Tensor(train_features), torch.FloatTensor(train_labels), torch.Tensor(val_features),
                torch.FloatTensor(val_labels))

    def _load_train_data(self):
        all_df = pd.read_csv(self._training_data,
                             sep='\t', header=None, names=['cell_line', 'smiles', 'auc', 'dataset'])

        train_cell_lines = list(set(all_df['cell_line']))
        val_cell_lines = []
        val_size = int(len(train_cell_lines) / 5)

        for _ in range(val_size):
            r = rd.randint(0, len(train_cell_lines) - 1)
            val_cell_lines.append(train_cell_lines.pop(r))

        val_df = all_df.query('cell_line in @val_cell_lines').reset_index(drop=True)
        train_df = all_df.query('cell_line in @train_cell_lines').reset_index(drop=True)

        std_df = util.calc_std_vals(train_df, self.zscore_method)
        std_df.to_csv(self.std, sep='\t', header=False, index=False)
        train_df = util.standardize_data(train_df, std_df)
        val_df = util.standardize_data(val_df, std_df)

        train_features = []
        train_labels = []
        for row in train_df.values:
            train_features.append([self.cell_id_mapping[row[0]]])
            train_labels.append([float(row[2])])

        val_features = []
        val_labels = []
        for row in val_df.values:
            val_features.append([self.cell_id_mapping[row[0]]])
            val_labels.append([float(row[2])])

        return train_features, val_features, train_labels, val_labels

    def _load_graph(self, file_name):

        digraph = nx.DiGraph()
        cx2factory = RawCX2NetworkFactory()
        nxfactory = CX2NetworkXFactory()
        digraph = nxfactory.get_graph(cx2factory.get_cx2network(file_name), digraph)
        roots = [n for n in digraph.nodes if digraph.in_degree(n) == 0]

        ugraph = digraph.to_undirected()
        connected_sub_graph_list = list(nxacc.connected_components(ugraph))

        print('There are', len(roots), 'roots:', roots[0])
        print('There are', len(digraph.nodes()), 'terms')
        print('There are', len(connected_sub_graph_list), 'connected componenets')

        if len(roots) > 1:
            print('There are more than 1 root of ontology. Please use only one root.')
            sys.exit(1)
        if len(connected_sub_graph_list) > 1:
            print('There are more than connected components. Please connect them.')
            sys.exit(1)

        self.digraph = digraph
        self.root = roots[0]
        # TODO: determine term_size_map and term_direct_gene_mapa
        # self.term_size_map = term_size_map
        # self.term_direct_gene_map = term_direct_gene_map
