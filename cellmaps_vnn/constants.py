"""
Contains constants used by cellmaps vnn
"""

IMPORTANCE_SCORE = 'importance_score'
"""
Importance score (set to P_rho currently)
"""

PRHO_SCORE = 'P_rho'
"""
P rho score
"""

P_PVAL_SCORE = 'P_pval'
"""
P pval score
"""

CRHO_SCORE = 'C_rho'
"""
C rho score
"""

C_PVAL_SCORE = 'C_pval'
"""
C pval score
"""

RLIPP_SCORE = 'RLIPP'
"""
RLIPP score
"""

RLIPP_OUTPUT_FILE = 'rlipp.out'
"""
Output file from rlipp algorithm
"""

GENE_RHO_FILE = 'gene_rho.out'
"""
Output file for gene Rho from rlipp algorithm
"""

HIERARCHY_FILENAME = 'hierarchy.cx2'
"""
Hierarchy filename.
"""

ORIGINAL_HIERARCHY_FILENAME = 'original_hierarchy.cx2'
"""
Original hierarchy filename.
"""

PARENT_NETWORK_NAME = 'hierarchy_parent.cx2'
"""
Parent network of hierarchy filename.
"""

TRAINING_DATA_FILENAME = 'training_data.txt'
"""
Default training data filename bundled with input directories.
"""

GENE2ID_FILENAME = 'gene2ind.txt'
"""
Default gene-to-index mapping filename bundled with input directories.
"""

CELL2ID_FILENAME = 'cell2ind.txt'
"""
Default cell-to-index mapping filename bundled with input directories.
"""

MUTATIONS_FILENAMES = ('cell2mutations.txt', 'cell2mutation.txt')
"""
Candidate filenames for mutation features bundled with input directories.
"""

CN_DELETIONS_FILENAMES = ('cell2cndeletion.txt',)
"""
Candidate filenames for copy-number deletion features bundled with input directories.
"""

CN_AMPLIFICATIONS_FILENAMES = ('cell2cnamplifications.txt', 'cell2cnamplification.txt')
"""
Candidate filenames for copy-number amplification features bundled with input directories.
"""

TRAIN_REQUIRED_INPUT_FILENAMES = {
    'training_data': (TRAINING_DATA_FILENAME,),
    'gene2id': (GENE2ID_FILENAME,),
    'cell2id': (CELL2ID_FILENAME,),
    'mutations': MUTATIONS_FILENAMES,
    'cn_deletions': CN_DELETIONS_FILENAMES,
    'cn_amplifications': CN_AMPLIFICATIONS_FILENAMES,
}
"""
Mapping of train argument names to the filenames that should be discovered in an input directory.
"""

GENE_SET_COLUMN_NAME = 'CD_MemberList'
"""
Name of the node attribute of the hierarchy with list of genes/ proteins of this node.
"""

GENE_SET_WITH_DATA = 'VNN_gene_set_with_data'
"""
Hierarchy node attribute that contain genes with available data (eg. mutation, deletion, amplification) for vnn model
"""

GENE_SET_SIZE = 'Gene_set_size'
"""
Size of gene set used for VNN
"""

SCORE_FILE_NAME_SUFFIX = '_gene_scores.out'
"""
Suffix for gene score file
"""

SYSTEM_INTERACTOME_FILE_SUFFIX = '_interactome.cx2'
"""
Suffix for system's interactome file name
"""

EDGE_IMPORTANCE_SCORE = 'edge_importance_score'
"""
Name of the edge importance score attribute
"""

MUTATION_IMPORTANCE_SCORE = 'mutation_importance_score'
DELETION_IMPORTANCE_SCORE = 'deletion_importance_score'
AMPLIFICATION_IMPORTANCE_SCORE = 'amplification_importance_score'
GENE_IMPORTANCE_SCORE = 'importance_score'
"""
Gene importance scores
"""

DEFAULT_BATCHSIZE = 64
DEFAULT_ZSCORE_METHOD = 'auc'
DEFAULT_GENOTYPE_HIDDENS = 4
DEFAULT_CUDA = 0
"""
Set of constants for VNNTrain and VNNPredict
"""
