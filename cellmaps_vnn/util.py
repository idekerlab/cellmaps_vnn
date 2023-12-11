import math
import os

import numpy as np
import pandas as pd
import torch
from torch._six import inf

from cellmaps_vnn.exceptions import CellmapsvnnError


def calc_std_vals(df, zscore_method):
    """
    TODO
    """
    std_df = pd.DataFrame(columns=['dataset', 'center', 'scale'])
    std_list = []

    if zscore_method == 'zscore':
        for name, group in df.groupby(['dataset'])['auc']:
            center = group.mean()
            scale = group.std()
            if math.isnan(scale) or scale == 0.0:
                scale = 1.0
            temp = pd.DataFrame([[name, center, scale]], columns=std_df.columns)
            std_list.append(temp)

    elif zscore_method == 'robustz':
        for name, group in df.groupby(['dataset'])['auc']:
            center = group.median()
            scale = group.quantile(0.75) - group.quantile(0.25)
            if math.isnan(scale) or scale == 0.0:
                scale = 1.0
            temp = pd.DataFrame([[name, center, scale]], columns=std_df.columns)
            std_list.append(temp)
    else:
        for name, group in df.groupby(['dataset'])['auc']:
            temp = pd.DataFrame([[name, 0.0, 1.0]], columns=std_df.columns)
            std_list.append(temp)

    std_df = pd.concat(std_list, ignore_index=True)
    return std_df


def standardize_data(df, std_df):
    """
    TODO
    """
    merged = pd.merge(df, std_df, how="left", on=['dataset'], sort=False)
    merged['z'] = (merged['auc'] - merged['center']) / merged['scale']
    merged = merged[['cell_line', 'smiles', 'z']]
    return merged


def load_numpy_data(file_path):
    """
    TODO
    """
    if not os.path.isfile(file_path):
        raise CellmapsvnnError(f"File {file_path} not found.")

    try:
        return np.genfromtxt(file_path, delimiter=',')
    except Exception as e:
        raise CellmapsvnnError(f"Error loading data from {file_path}: {e}")


def load_mapping(mapping_file, mapping_type):
    """
    TODO
    """
    if not os.path.isfile(mapping_file):
        raise CellmapsvnnError(f"Mapping file {mapping_file} not found.")

    mapping = {}
    file_handle = open(mapping_file)
    for line in file_handle:
        line = line.rstrip().split()
        mapping[line[1]] = int(line[0])

    file_handle.close()
    print('Total number of {} = {}'.format(mapping_type, len(mapping)))
    return mapping


# build mask: matrix (nrows = number of relevant gene set, ncols = number all genes)
# elements of matrix are 1 if the corresponding gene is one of the relevant genes
def create_term_mask(term_direct_gene_map, gene_dim, cuda_id):
    term_mask_map = {}
    for term, gene_set in term_direct_gene_map.items():
        mask = torch.zeros(len(gene_set), gene_dim).cuda(cuda_id)
        for i, gene_id in enumerate(gene_set):
            mask[i, gene_id] = 1
        term_mask_map[term] = mask
    return term_mask_map


def build_input_vector(input_data, cell_features):
    genedim = len(cell_features[0, :])
    featdim = len(cell_features[0, 0, :])
    feature = np.zeros((input_data.size()[0], genedim, featdim))

    for i in range(input_data.size()[0]):
        feature[i] = cell_features[int(input_data[i, 0])]

    feature = torch.from_numpy(feature).float()
    return feature


def get_grad_norm(model_params, norm_type):
    """Gets gradient norm of an iterable of model_params.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Arguments:
        model_params (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
    Returns: Total norm of the model_params (viewed as a single vector).
    """
    if isinstance(model_params, torch.Tensor):  # check if parameters are tensorobject
        model_params = [model_params]  # change to list
    model_params = [p for p in model_params if p.grad is not None]  # get list of params with grads
    norm_type = float(norm_type)  # make sure norm_type is of type float
    if len(model_params) == 0:  # if no params provided, return tensor of 0
        return torch.tensor(0.)

    device = model_params[0].grad.device  # get device
    if norm_type == inf:  # infinity norm
        total_norm = max(p.grad.detach().abs().max().to(device) for p in model_params)
    else:  # total norm
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in model_params]),
                                norm_type)
    return total_norm


def pearson_corr(x, y):
    xx = x - torch.mean(x)
    yy = y - torch.mean(y)

    return torch.sum(xx * yy) / (torch.norm(xx, 2) * torch.norm(yy, 2))
