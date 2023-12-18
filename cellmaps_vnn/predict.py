import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.utils.data as du
from torch.autograd import Variable
from cellmaps_utils import constants

from cellmaps_vnn import util
from cellmaps_vnn.exceptions import CellmapsvnnError

logger = logging.getLogger(__name__)


class VNNPredict:
    COMMAND = 'predict'

    def __init__(self, theargs):
        """
        Constructor for predicting with a trained model.
        """
        self._theargs = theargs

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
        parser.add_argument('outdir', help='Directory to write results to')
        parser.add_argument('--modeldir', required=True, help='Path to RO-Crate with the trained model', type=str)
        parser.add_argument('--predict_data', required=True, help='Path to the dataset to be predicted', type=str)
        parser.add_argument('--gene2id', help='Gene to ID mapping file', type=str)
        parser.add_argument('--cell2id', required=True, help='Cell to ID mapping file', type=str)
        parser.add_argument('--mutations', required=True, help='Mutation information for cell lines', type=str)
        parser.add_argument('--cn_deletions', required=True, help='Copy number deletions for cell lines', type=str)
        parser.add_argument('--cn_amplifications', required=True, help='Copy number amplifications for cell lines',
                            type=str)
        parser.add_argument('--batchsize', help='Batchsize', type=int, default=1000)
        parser.add_argument('--cuda', help='Specify GPU', type=int, default=0)
        parser.add_argument('--zscore_method', help='zscore method (zscore/robustz)', type=str, default='auc')
        parser.add_argument('--std', help='Standardization File (if not set standardization file from RO-Crate '
                                          'will be used)', type=str)
        return parser

    def run(self):
        """
        The logic for running predictions with the model.
        """
        try:
            model = os.path.join(self._theargs.modeldir, 'model_final.pt')
            std = os.path.join(self._theargs.modeldir, 'std.txt') if self._theargs.std is None else self._theargs.std
            torch.set_printoptions(precision=5)

            # Load data and model for prediction
            predict_data, cell2id_mapping = self._prepare_predict_data(
                self._theargs.predict_data, self._theargs.cell2id, self._theargs.zscore_method, std)

            # Load cell features
            cell_features = util.load_cell_features(self._theargs.mutations, self._theargs.cn_deletions,
                                                    self._theargs.cn_amplifications)

            result_file_prefix = os.path.join(self._theargs.outdir, 'predict')
            hidden_dir = os.path.join(self._theargs.outdir, 'hidden/')
            if not os.path.exists(hidden_dir):
                os.mkdir(hidden_dir)

            # Perform prediction
            self.predict(predict_data, model, hidden_dir, self._theargs.batchsize,
                         result_file_prefix, cell_features)

        except Exception as e:
            logger.error(f"Error in prediction flow: {e}")
            raise CellmapsvnnError(f"Encountered problem in prediction flow: {e}")

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

    def predict(self, predict_data, model_file, hidden_folder, batch_size, result_file='predict', cell_features=None):
        """
        Perform prediction using the trained model.

        :param predict_data: Tuple of features and labels for prediction.
        :param model_file: Path to the trained model file.
        :param hidden_folder: Directory to store hidden layer outputs.
        :param batch_size: Size of each batch for prediction.
        :param result_file: File path prefix for storing prediction results.
        :param cell_features: Additional cell features for prediction.
        """
        try:
            model = self._load_model(model_file)
            test_loader = self._create_data_loader(predict_data, batch_size)
            test_predict, saved_grads = self._predict(model, test_loader, cell_features, hidden_folder, result_file)

            predict_label_gpu = predict_data[1].cuda(self._theargs.cuda)
            test_corr = util.pearson_corr(test_predict, predict_label_gpu)
            logger.info(f"Test correlation {model.root}: {test_corr:.4f}")

            np.savetxt(f'{result_file}.txt', test_predict.cpu().numpy(), '%.4e')

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise CellmapsvnnError(f"Encountered problem in prediction: {e}")

    def _load_model(self, model_file):
        """
        Load the trained model for prediction.

        :param model_file: Path to the trained model file.
        :return: Loaded model.
        """
        model = torch.load(model_file, map_location=f'cuda:{self._theargs.cuda}')
        model.cuda(self._theargs.cuda)
        model.eval()
        return model

    def _create_data_loader(self, predict_data, batch_size):
        """
        Create a DataLoader for the prediction data.

        :param predict_data: Tuple of features and labels for prediction.
        :param batch_size: Size of each batch for prediction.
        :return: DataLoader for the prediction data.
        """
        predict_feature, predict_label = predict_data
        return du.DataLoader(du.TensorDataset(predict_feature, predict_label), batch_size=batch_size, shuffle=False)

    def _predict(self, model, data_loader, cell_features, hidden_folder, result_file):
        """
        Run the prediction process and save outputs.

        :param model: Trained model for prediction.
        :param data_loader: DataLoader containing the prediction data.
        :param cell_features: Additional cell features for prediction.
        :param hidden_folder: Directory to store hidden layer outputs.
        :param result_file: File path prefix for storing prediction results.
        :return: Tuple of prediction results and saved gradients.
        """
        test_predict = torch.zeros(0, 0).cuda(self._theargs.cuda)
        saved_grads = {}

        for i, (inputdata, labels) in enumerate(data_loader):
            cuda_features = self._process_input(inputdata, cell_features)
            aux_out_map, hidden_embeddings_map = model(cuda_features)
            test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0) \
                if test_predict.size()[0] else aux_out_map['final'].data

            self._save_hidden_outputs(hidden_embeddings_map, hidden_folder)
            self._register_gradient_hooks(hidden_embeddings_map, saved_grads)
            self._backpropagate(aux_out_map)

            self._save_gradients(cuda_features, result_file)
            self._save_hidden_gradients(saved_grads, hidden_folder)

        return test_predict, saved_grads

    def _process_input(self, inputdata, cell_features):
        """
        Process input data for the model.

        :param inputdata: Input data for the model.
        :param cell_features: Additional cell features for prediction.
        :return: Processed features as CUDA variables.
        """
        features = util.build_input_vector(inputdata, cell_features)
        return Variable(features.cuda(self._theargs.cuda), requires_grad=True)

    def _save_hidden_outputs(self, hidden_embeddings_map, hidden_folder):
        """
        Save outputs from hidden layers.

        :param hidden_embeddings_map: Dictionary of hidden layer outputs.
        :param hidden_folder: Directory to save hidden layer outputs.
        """
        for element, hidden_map in hidden_embeddings_map.items():
            hidden_file = os.path.join(hidden_folder, element + '.hidden')
            with open(hidden_file, 'ab') as f:
                np.savetxt(f, hidden_map.data.cpu().numpy(), '%.4e')

    def _register_gradient_hooks(self, hidden_embeddings_map, saved_grads):
        """
        Register gradient hooks to save gradients of hidden layers.

        :param hidden_embeddings_map: Dictionary of hidden layer outputs.
        :param saved_grads: Dictionary to store saved gradients.
        """

        def save_grad(elem):
            def savegrad_hook(grad):
                saved_grads[elem] = grad

            return savegrad_hook

        for element, _ in hidden_embeddings_map.items():
            hidden_embeddings_map[element].register_hook(save_grad(element))

    def _backpropagate(self, aux_out_map):
        """
        Perform backpropagation.

        :param aux_out_map: Auxiliary output map from the model.
        """
        aux_out_map['final'].backward(torch.ones_like(aux_out_map['final']))

    def _save_gradients(self, cuda_features, result_file):
        """
        Save gradients for each feature.

        :param cuda_features: CUDA features variable.
        :param result_file: File path prefix for storing gradients.
        """
        for i in range(len(cuda_features[0, 0, :])):
            feature_grad = cuda_features.grad.data[:, :, i]
            grad_file = f'{result_file}_feature_grad_{i}.txt'
            with open(grad_file, 'ab') as f:
                np.savetxt(f, feature_grad.cpu().numpy(), '%.4e', delimiter='\t')

    def _save_hidden_gradients(self, saved_grads, hidden_folder):
        """
        Save the gradients of the hidden layer outputs.

        :param saved_grads: Dictionary containing the saved gradients.
        :param hidden_folder: Directory to save the hidden layer gradients.
        """
        for element, hidden_grad in saved_grads.items():
            hidden_grad_file = os.path.join(hidden_folder, f'{element}.hidden_grad')
            with open(hidden_grad_file, 'ab') as f:
                np.savetxt(f, hidden_grad.data.cpu().numpy(), '%.4e', delimiter='\t')
