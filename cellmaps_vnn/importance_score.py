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

    def __init__(self, outdir, hierarchy):
        super().__init__(outdir=outdir, hierarchy=hierarchy)

    def calc_scores(self):
        for node, node_val in self._hierarchy.get_nodes().items():
            members = node_val.get('v', {}).get(constants.GENE_SET_WITH_DATA, None)
            if members is not None:
                file_path = os.path.join(self._outdir, str(node) + constants.SCORE_FILE_NAME_SUFFIX)
                with open(file_path, 'w') as score_file:
                    score_file.write('gene\tmutation_importance_score\tdeletion_importance_score'
                                     '\tamplification_importance_score\timportance_score\n')
                    for m in members:
                        score_file.write(f'{m}\t{round(random.random(), 3)}\t{round(random.random(), 3)}\t'
                                         f'{round(random.random(), 3)}\t{round(random.random(), 3)}\n')


