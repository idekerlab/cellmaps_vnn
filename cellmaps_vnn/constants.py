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

GENE_SET_COLUMN_NAME = 'CD_MemberList'
"""
Name of the node attribute of the hierarchy with list of genes/ proteins of this node.
"""

GENE_SET_WITH_DATA = 'VNN_gene_set_with_data'
"""
Hierarchy node attribute that contain genes with available data (eg. mutation, deletion, amplification) for vnn model
"""

DEFAULT_BATCHSIZE = 64
DEFAULT_ZSCORE_METHOD = 'auc'
DEFAULT_GENOTYPE_HIDDENS = 4
DEFAULT_CUDA = 0
"""
Set of constants for VNNTrain and VNNPredict
"""
