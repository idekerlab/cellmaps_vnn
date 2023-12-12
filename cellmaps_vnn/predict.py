import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.utils.data as du
from torch.autograd import Variable
from cellmaps_utils import constants

from cellmaps_vnn import util

logger = logging.getLogger(__name__)


class VNNPredict:
    COMMAND = 'predict'

    def __init__(self, theargs):
        """
        Constructor for predicting with a trained model.
        """
        self._theargs = theargs
        # TODO: Initialize other necessary variables or configurations

    @staticmethod
    def add_subparser(subparsers):
        """
        Adds a subparser for the 'predict' command.
        """
        # TODO: modify description later
        desc = """
        Version: todo

        The 'predict' command takes a trained model and input data to run predictions.
        The results are stored in a specified output directory.
        """
        parser = subparsers.add_parser(VNNPredict.COMMAND,
                                       help='Run prediction using a trained model',
                                       description=desc,
                                       formatter_class=constants.ArgParseFormatter)
        parser.add_argument('--predict_data', required=True, help='Path to the dataset to be predicted', type=str)
        parser.add_argument('--model', required=True, help='Path to the trained model in RO-Crate', type=str)
        parser.add_argument('--batchsize', help='Batchsize', type=int, default=1000)
        parser.add_argument('--gene2id', help='Gene to ID mapping file', type=str)
        parser.add_argument('--cell2id', help='Cell to ID mapping file', type=str)
        parser.add_argument('--hidden', help='Hidden output folder', type=str, default='hidden/')
        parser.add_argument('--result', help='Result file prefix', type=str, default='result/predict')
        parser.add_argument('--cuda', help='Specify GPU', type=int, default=0)
        parser.add_argument('--mutations', help='Mutation information for cell lines', type=str)
        parser.add_argument('--cn_deletions', help='Copy number deletions for cell lines', type=str)
        parser.add_argument('--cn_amplifications', help='Copy number amplifications for cell lines', type=str)
        parser.add_argument('--zscore_method', help='zscore method (zscore/robustz)', type=str)
        parser.add_argument('--std', help='Standardization File', type=str)
        # TODO: Add other necessary arguments - shall common arguments be extracted to cmd file
        return parser

    def run(self):
        """
        The logic for running predictions with the model.
        """
        try:
            torch.set_printoptions(precision=5)

            # Load data and model for prediction
            predict_data, cell2id_mapping = self._prepare_predict_data(
                self._theargs.predict_data, self._theargs.cell2id, self._theargs.zscore_method, self._theargs.std)
            num_genes = len(util.load_mapping(self._theargs.gene2id, "genes"))

            # Load cell features
            cell_features = util.load_cell_features(self._theargs.mutations, self._theargs.cn_deletions,
                                                    self._theargs.cn_amplifications)

            # Perform prediction
            self.predict(predict_data, num_genes, self._theargs.model, self._theargs.hidden, self._theargs.batchsize,
                         self._theargs.result, cell_features)

        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise

    def _prepare_predict_data(self, test_file, cell2id_mapping_file, zscore_method, std_file):
        cell2id_mapping = util.load_mapping(cell2id_mapping_file, 'cell lines')
        test_features, test_labels = self._load_pred_data(test_file, cell2id_mapping, zscore_method, std_file)
        return (torch.Tensor(test_features), torch.Tensor(test_labels)), cell2id_mapping

    @staticmethod
    def _load_pred_data(test_file, cell2id, zscore_method, train_std_file):

        train_std_df = pd.read_csv(train_std_file, sep='\t', header=None, names=['dataset', 'center', 'scale'])
        test_df = pd.read_csv(test_file, sep='\t', header=None, names=['cell_line', 'smiles', 'auc', 'dataset'])
        test_std_df = util.calc_std_vals(test_df, zscore_method)
        for i, row in test_std_df.iterrows():
            dataset = row['dataset']
            train_entry = train_std_df.query('dataset == @dataset')
            if not train_entry.empty:
                test_std_df.loc[i, 'center'] = float(train_entry['center'])
                test_std_df.loc[i, 'scale'] = float(train_entry['scale'])
        test_df = util.standardize_data(test_df, test_std_df)

        feature = []
        label = []
        for row in test_df.values:
            feature.append([cell2id[row[0]]])
            label.append([float(row[2])])
        return feature, label

    def predict(self, predict_data, gene_dim, model_file, hidden_folder, batch_size, result_file, cell_features):
        feature_dim = gene_dim

        model = torch.load(model_file, map_location='cuda:%d' % self._theargs.cuda)
        predict_feature, predict_label = predict_data
        predict_label_gpu = predict_label.cuda(self._theargs.cuda)

        model.cuda(self._theargs.cuda)
        model.eval()

        test_loader = du.DataLoader(du.TensorDataset(predict_feature, predict_label), batch_size=batch_size,
                                    shuffle=False)

        test_predict = torch.zeros(0, 0).cuda(self._theargs.cuda)
        hidden_embeddings_map = {}

        saved_grads = {}

        def save_grad(element):
            def savegrad_hook(grad):
                saved_grads[element] = grad

            return savegrad_hook

        for i, (inputdata, labels) in enumerate(test_loader):
            features = util.build_input_vector(inputdata, cell_features)
            cuda_features = Variable(features.cuda(self._theargs.cuda), requires_grad=True)

            aux_out_map, hidden_embeddings_map = model(cuda_features)

            if test_predict.size()[0] == 0:
                test_predict = aux_out_map['final'].data
            else:
                test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)

            for element, hidden_map in hidden_embeddings_map.items():
                hidden_file = os.path.join(hidden_folder, element + '.hidden')
                with open(hidden_file, 'ab') as f:
                    np.savetxt(f, hidden_map.data.cpu().numpy(), '%.4e')

            for element, _ in hidden_embeddings_map.items():
                hidden_embeddings_map[element].register_hook(save_grad(element))

            aux_out_map['final'].backward(torch.ones_like(aux_out_map['final']))

            feature_grad = torch.zeros(0, 0).cuda(self._theargs.cuda)
            for i in range(len(cuda_features[0, 0, :])):
                feature_grad = cuda_features.grad.data[:, :, i]
                grad_file = result_file + '_feature_grad_' + str(i) + '.txt'
                with open(grad_file, 'ab') as f:
                    np.savetxt(f, feature_grad.cpu().numpy(), '%.4e', delimiter='\t')

            for element, hidden_grad in saved_grads.items():
                hidden_grad_file = os.path.join(hidden_folder, element + '.hidden_grad')
                with open(hidden_grad_file, 'ab') as f:
                    np.savetxt(f, hidden_grad.data.cpu().numpy(), '%.4e', delimiter='\t')

        test_corr = util.pearson_corr(test_predict, predict_label_gpu)
        print("Test correlation\t%s\t%.4f" % (model.root, test_corr))

        np.savetxt(result_file + '.txt', test_predict.cpu().numpy(), '%.4e')
