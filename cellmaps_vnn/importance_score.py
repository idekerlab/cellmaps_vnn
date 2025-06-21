import os
import random

from cellmaps_vnn import constants


class ImportanceScoreCalculator(object):
    """
    """

    def __init__(self, outdir=None, hierarchy=None):
        """
        Constructor
        """
        self._outdir = outdir
        self._hierarchy = hierarchy

    def calc_scores(self):
        raise NotImplementedError('Subclasses should implement')


class FakeGeneImportanceScoreCalculator(ImportanceScoreCalculator):
    SCORE_FILE_BASE_NAME = 'gene_scores_system_'

    def __init__(self, outdir, hierarchy):
        super().__init__(outdir=outdir, hierarchy=hierarchy)
        self._score_file_base = os.path.join(self._outdir, self.SCORE_FILE_BASE_NAME)

    def calc_scores(self):
        for node, node_val in self._hierarchy.get_nodes().items():
            members = node_val.get('v', {}).get(constants.GENE_SET_WITH_DATA, None)
            if members is not None:
                file_path = self._score_file_base + str(node) + '.out'
                with open(file_path, 'w') as score_file:
                    score_file.write('gene\tmutation_importance_score\tdeletion_importance_score'
                                     '\tamplification_importance_score\timportance_score\n')
                    for m in members:
                        score_file.write(f'{m}\t{round(random.random(), 3)}\t{round(random.random(), 3)}\t'
                                         f'{round(random.random(), 3)}\t{round(random.random(), 3)}\n')


