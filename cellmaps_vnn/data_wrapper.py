import sys
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import numpy as np
from ndex2.cx2 import RawCX2NetworkFactory, CX2NetworkXFactory


class DataWrapper:

    def __init__(self, args):

        self.root = None
        self.dG = None
        self.num_hiddens_genotype = args.genotype_hiddens
        self.lr = args.lr
        self.wd = args.wd
        self.alpha = args.alpha
        self.epochs = args.epoch
        self.batchsize = args.batchsize
        self.modeldir = args.modeldir
        self.cuda = args.cuda
        self.patience = args.patience
        self.delta = args.delta
        self.min_dropout_layer = args.min_dropout_layer
        self.dropout_fraction = args.dropout_fraction
        # self.mutations = np.genfromtxt(args.mutations, delimiter=',')
        # self.cn_deletions = np.genfromtxt(args.cn_deletions, delimiter=',')
        # self.cn_amplifications = np.genfromtxt(args.cn_amplifications, delimiter=',')
        # self.cell_features = np.dstack([self.mutations, self.cn_deletions, self.cn_amplifications])
        # self.train_feature, self.train_label, self.val_feature, self.val_label = self.prepare_train_data()
        # self.cell_id_mapping = util.load_mapping(args.cell2id, 'cell lines')
        # self.gene_id_mapping = util.load_mapping(args.gene2id, 'genes')
        self.train = args.training_data
        # self.load_graph(args.hierarchy)

    def load_graph(self, file_name):

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

        self.dG = digraph
        self.root = roots[0]
        # TODO: determine term_size_map and term_direct_gene_mapa
        # self.term_size_map = term_size_map
        # self.term_direct_gene_map = term_direct_gene_map
