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

    def __init__(self, hierarchy):
        super().__init__(hierarchy=hierarchy)

    def calculate_gene_importance_scores(self):
        # for node in self._hierarchy.get_nodes().items():
        #     members = node.get('v', {}).get(self._gene_column).split(' ')
        pass
