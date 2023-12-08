import math
import os

import numpy as np
import pandas as pd

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
