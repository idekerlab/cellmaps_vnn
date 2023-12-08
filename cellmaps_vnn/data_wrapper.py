import sys
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
        self.mutations = util.load_numpy_data(theargs.mutations)
        self.cn_deletions = util.load_numpy_data(theargs.cn_deletions)
        self.cn_amplifications = util.load_numpy_data(theargs.cn_amplifications)

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

        try:
            digraph = self._create_digraph(file_name)
            roots = [n for n in digraph.nodes if digraph.in_degree(n) == 0]
            ugraph = digraph.to_undirected()
            connected_sub_graph_list = list(nx.connected_components(ugraph))

            if len(roots) != 1 or len(connected_sub_graph_list) != 1:
                raise CellmapsvnnError("Graph must have exactly one root and be fully connected")

            self.root = roots[0]
            # TODO: determine term_size_map and term_direct_gene_map
            # self.term_size_map = term_size_map
            # self.term_direct_gene_map = term_direct_gene_map

        except Exception as e:
            print("Error loading graph:", e)
            sys.exit(1)

    def _create_digraph(self, file_name):
        cx2factory = RawCX2NetworkFactory()
        nxfactory = CX2NetworkXFactory()
        digraph = nxfactory.get_graph(cx2factory.get_cx2network(file_name), nx.DiGraph())
        self.digraph = digraph
        return digraph
