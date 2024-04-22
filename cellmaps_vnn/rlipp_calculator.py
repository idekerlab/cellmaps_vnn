import numpy as np
import pandas as pd
import time
from scipy import stats
from multiprocessing import Pool
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV


class RLIPPCalculator:
    """
    A calculator for Relative Importance of Predictor Performance (RLIPP) scores.

    Parameters:
    outdir (str): Output directory for the RLIPP scores and gene correlations.
    hierarchy (CX2Network): A hierarchy in HCX format.
    test_data (str): Path to the CSV file containing test data.
    predicted_data (str): Path to the file containing predicted values.
    gene2idfile (str): Path to the file mapping genes to IDs.
    cell2idfile (str): Path to the file mapping cells to IDs.
    hidden_dir (str): Directory containing hidden layer outputs.
    rlipp_file (str): Path of the output file where results of rlipp algorithm will be saved
    gene_rho_file (str): Path of the output file where gene rho scores will be saved
    cpu_count (int): No of available cores
    num_hiddens_genotype (int): Mapping for the number of neurons in each term in genotype parts
    drug_count (int): No of top performing drugs
    """
    def __init__(self, hierarchy, test_data, predicted_data, gene2idfile, cell2idfile, hidden_dir,
                 rlipp_file, gene_rho_file, cpu_count, num_hiddens_genotype, drug_count):
        self._hierarchy = hierarchy
        self.terms = list(hierarchy.get_nodes().keys())
        self.test_df = pd.read_csv(test_data, sep='\t', header=None, names=['C', 'D', 'AUC', 'DS'])
        self.predicted_vals = np.loadtxt(predicted_data)
        self.genes = pd.read_csv(gene2idfile, sep='\t', header=None, names=['I', 'G'])['G']
        self.cell_index = pd.read_csv(cell2idfile, sep="\t", header=None, names=['I', 'C'])
        self.hidden_dir = hidden_dir
        self.rlipp_file = rlipp_file
        self.gene_rho_file = gene_rho_file
        self.cpu_count = cpu_count
        self.num_hiddens_genotype = num_hiddens_genotype
        self.drug_count = drug_count
        self.drugs = list(set(self.test_df['D']))
        if self.drug_count == 0:
            self.drug_count = len(self.drugs)
    #     self.pool = Pool(cpu_count)
    #
    # def __del__(self):
    #     """
    #     Ensure Pool is properly closed when the object is deleted
    #     """
    #     self.pool.close()

    def create_drug_pos_map(self):
        """
        Creates a mapping from drugs to their positions in the test data file.

        :return: A dictionary where keys are drugs and values are lists of positions in the test data.
        :rtype: dict
        """
        drug_pos_map = {}
        for i, row in self.test_df.iterrows():
            if row['D'] not in drug_pos_map:
                drug_pos_map[row['D']] = []
            drug_pos_map[row['D']].append(i)
        return drug_pos_map

    def create_drug_corr_map_sorted(self, drug_pos_map):
        """
        Creates a sorted mapping of drugs to their Spearman correlation values.

        :param drug_pos_map: A dictionary mapping drugs to their positions in the test data.
        :type drug_pos_map: dict

        :return: A dictionary of drugs sorted by their Spearman correlation values in descending order.
        :rtype: dict
        """
        drug_corr_map = {}
        for d in self.drugs:
            if len(drug_pos_map[d]) == 0:
                drug_corr_map[d] = 0.0
                continue
            test_vals = np.take(np.array(self.test_df['AUC']), drug_pos_map[d])
            pred_vals = np.take(self.predicted_vals, drug_pos_map[d])
            drug_corr_map[d] = stats.spearmanr(test_vals, pred_vals)[0]
        return {drug: corr for drug, corr in sorted(drug_corr_map.items(), key=lambda item: item[1], reverse=True)}

    def load_feature(self, element, size):
        """
        Loads hidden features for a given element.

        :param element: The element (term or gene) whose features are to be loaded.
        :type element: str
        :param size: The number of columns (features) to load.
        :type size: int

        :return: A numpy array of the hidden features for the given element.
        :rtype: numpy.ndarray
        """
        file_name = self.hidden_dir + "/" + str(element) + '.hidden'
        return np.loadtxt(file_name, usecols=range(size))

    def load_term_features(self, term):
        """
        Loads hidden features for a given term.

        :param term: The term whose features are to be loaded.
        :type term: str

        :return: A numpy array of the hidden features for the given term.
        :rtype: numpy.ndarray
        """
        return self.load_feature(term, self.num_hiddens_genotype)

    def load_gene_features(self, gene):
        """
        Loads hidden features for a given gene.

        :param gene: The gene whose features are to be loaded.
        :type gene: str

        :return: A numpy array of the hidden features for the given gene.
        :rtype: numpy.ndarray
        """
        return self.load_feature(gene, 1)

    def create_child_feature_map(self, feature_map, term):
        """
        Creates a map of child features for a given term.

        :param feature_map: A dictionary mapping terms/genes to their features.
        :type feature_map: dict
        :param term: The term for which child features are to be created.
        :type term: str

        :return: A list of child features for the given term.
        :rtype: list
        """
        child_features = [term]
        child_features.extend(
            feature_map[edge_data['t']] for _, edge_data in self._hierarchy.get_edges().items() if
            edge_data['s'] == term)
        return child_features

    def load_all_features(self):
        """
        Loads hidden features for all terms and genes.

        :return: A tuple containing two dictionaries, one mapping terms/genes to their features and the other mapping
                 terms to their child features.
        :rtype: (dict, dict)
        """
        feature_map = {}

        # Load term features
        with Pool(self.cpu_count) as p:
            results = p.map(self.load_term_features, self.terms)
        for i, term in enumerate(self.terms):
            feature_map[term] = results[i]

        # Load gene features
        with Pool(self.cpu_count) as p:
            results = p.map(self.load_gene_features, self.genes)
        for i, gene in enumerate(self.genes):
            feature_map[gene] = results[i]

        child_feature_map = {term: [] for term in self.terms}

        for term in self.terms:
            children = [edge_data['t'] for edge_id, edge_data in self._hierarchy.get_edges().items() if
                        edge_data['s'] == term]
            for child in children:
                if child in feature_map:
                    child_feature_map[term].append(feature_map[child])

        return feature_map, child_feature_map

    @staticmethod
    def get_child_features(term_child_features, position_map):
        """
        Gets a matrix of hidden features for a given term's children.

        :param term_child_features: A list of features for the children of a term.
        :type term_child_features: list
        :param position_map: A list of positions for which features are to be extracted.
        :type position_map: list

        :return: A matrix of hidden features for the children of the given term.
        :rtype: numpy.ndarray
        """
        child_features = []
        for f in term_child_features:
            child_features.append(np.take(f, position_map, axis=0))
        return np.column_stack([f for f in child_features])

    def exec_lm(self, X, y):
        """
        Executes 5-fold cross-validated Ridge regression for a given hidden features matrix
        and returns the Spearman correlation value of the predicted output.

        :param X: The input matrix for regression.
        :type X: numpy.ndarray
        :param y: The target variable.
        :type y: numpy.ndarray

        :return: A tuple containing the Spearman correlation coefficient and p-value.
        :rtype: (float, float)
        """
        pca = PCA(n_components=self.num_hiddens_genotype)
        X_pca = pca.fit_transform(X)

        regr = RidgeCV(cv=5)
        regr.fit(X_pca, y)
        y_pred = regr.predict(X_pca)
        return stats.spearmanr(y_pred, y)

    def calc_term_rlipp(self, term_features, term_child_features, position_map, term, drug):
        """
        Calculates the RLIPP score for a given term and drug.

        :param term_features: The features for the parent term.
        :type term_features: numpy.ndarray
        :param term_child_features: The features for the children of the term.
        :type term_child_features: list
        :param position_map: A list of positions for which RLIPP is to be calculated.
        :type position_map: list
        :param term: The term for which RLIPP is calculated.
        :type term: str
        :param drug: The drug for which RLIPP is calculated.
        :type drug: str

        :return: A formatted string containing the term, Spearman correlations, p-values, and RLIPP score.
        :rtype: str
        """
        if not term_child_features:
            return ''
        X_parent = np.take(term_features, position_map, axis=0)
        X_child = self.get_child_features(term_child_features, position_map)
        y = np.take(self.predicted_vals, position_map)
        p_rho, p_pval = self.exec_lm(X_parent, y)
        c_rho, c_pval = self.exec_lm(X_child, y)
        rlipp = p_rho / c_rho
        result = '{}\t{:.3e}\t{:.3e}\t{:.3e}\t{:.3e}\t{:.3e}\n'.format(term, p_rho, p_pval, c_rho, c_pval, rlipp)
        return result

    def calc_gene_rho(self, gene_features, position_map, gene, drug):
        """
        Calculates Spearman correlation between gene embeddings and predicted AUC.

        :param gene_features: The features for the gene.
        :type gene_features: numpy.ndarray
        :param position_map: A list of positions for which correlation is to be calculated.
        :type position_map: list
        :param gene: The gene for which correlation is calculated.
        :type gene: str
        :param drug: The drug for which correlation is calculated.
        :type drug: str

        :return: A formatted string containing the gene, Spearman correlation, and p-value.
        :rtype: str
        """
        pred = np.take(self.predicted_vals, position_map)
        gene_embeddings = np.take(gene_features, position_map)
        rho, p_val = stats.spearmanr(pred, gene_embeddings)
        result = '{}\t{:.3e}\t{:.3e}\n'.format(gene, rho, p_val)
        return result

    def calc_scores(self):
        """
        Calculates RLIPP scores for top n drugs (n = drug_count),
        and prints the result in "Drug Term P_rho C_rho RLIPP" format.

        This method runs the calculation in parallel for efficiency.
        """
        print('Starting score calculation')
        drug_pos_map = self.create_drug_pos_map()
        sorted_drugs = list(self.create_drug_corr_map_sorted(drug_pos_map).keys())[0:self.drug_count]

        start = time.time()
        feature_map, child_feature_map = self.load_all_features()
        print('Time taken to load features: {:.4f}'.format(time.time() - start))

        with open(self.rlipp_file, "w") as rlipp_file, open(self.gene_rho_file, "w") as gene_rho_file:
            rlipp_file.write('Term\tP_rho\tP_pval\tC_rho\tC_pval\tRLIPP\n')
            gene_rho_file.write('Gene\tRho\tP_val\n')

            with Parallel(backend="multiprocessing", n_jobs=self.cpu_count) as parallel:
                for i, drug in enumerate(sorted_drugs):
                    start = time.time()

                    # Parallel computation of RLIPP and gene correlation results
                    rlipp_results = parallel(
                        delayed(self.calc_term_rlipp)(feature_map[term], child_feature_map[term], drug_pos_map[drug], term,
                                                      drug) for term in self.terms)
                    gene_rho_results = parallel(
                        delayed(self.calc_gene_rho)(feature_map[gene], drug_pos_map[drug], gene, drug) for gene in
                        self.genes)

                    # After collecting all results, write them to files
                    for result in rlipp_results:
                        rlipp_file.write(result)
                    for result in gene_rho_results:
                        gene_rho_file.write(result)

                print('Drug {} completed in {:.4f} seconds'.format((i + 1), (time.time() - start)))