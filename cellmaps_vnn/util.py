import math

import pandas as pd


def calc_std_vals(df, zscore_method):
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
    merged = pd.merge(df, std_df, how="left", on=['dataset'], sort=False)
    merged['z'] = (merged['auc'] - merged['center']) / merged['scale']
    merged = merged[['cell_line', 'smiles', 'z']]
    return merged


def load_mapping(mapping_file, mapping_type):
    mapping = {}
    file_handle = open(mapping_file)
    for line in file_handle:
        line = line.rstrip().split()
        mapping[line[1]] = int(line[0])

    file_handle.close()
    print('Total number of {} = {}'.format(mapping_type, len(mapping)))
    return mapping
