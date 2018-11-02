import os

is_dev_mode = True

#### PATHS ####
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# data paths
DATA_DEV_MODE_PATH = os.path.join(ROOT_DIR, '../../../overnight_delex_dev')
DATA_TEST_MODE_PATH = os.path.join(ROOT_DIR, '../../../overnight_delex')
DATA_PATH = DATA_DEV_MODE_PATH if is_dev_mode else DATA_TEST_MODE_PATH
DATA_DEV_MODE_CROSS_PATH = os.path.join(ROOT_DIR, '../../../overnight_delex_dev/cross_domain')
DATA_TEST_MODE_CROSS_PATH = os.path.join(ROOT_DIR, '../../../overnight_delex/cross_domain')
DATA_CROSS_PATH = DATA_DEV_MODE_CROSS_PATH if is_dev_mode else DATA_TEST_MODE_CROSS_PATH

# executor
EVALUATOR_PATH = os.path.join(ROOT_DIR, '../evaluator/overnight')

# aligner model
MODEL_OUT_PATH = os.path.join(ROOT_DIR, '../../../models-align/')

# delexicalizer paths
LEX_PATH = os.path.join(ROOT_DIR, '../../../lexicons/')
DATA_RAW_PATH = os.path.join(ROOT_DIR, '../../../overnight-lf')
EMBEDDING_PATH = os.path.join(ROOT_DIR, '../../../data/glove/glove.840B.300d.filter.txt')
NLP_PATH = os.path.join(ROOT_DIR, '../../../stanford-corenlp-full-2016-10-31/')
JJ_PATH = os.path.join(DATA_PATH, 'jj_dist/jj_dist.txt')

# fast_align paths
ALIGNMENTS_IN_PATH = os.path.join(ROOT_DIR, '../../../fast_align_alignments_dev/input') if is_dev_mode else os.path.join(ROOT_DIR, '../../../fast_align_alignments/input')
ALIGNMENTS_OUT_PATH = os.path.join(ROOT_DIR, '../../../fast_align_alignments_dev/output') if is_dev_mode else os.path.join(ROOT_DIR, '../../../fast_align_alignments/output')
FAST_ALIGN_PATH = os.path.join(ROOT_DIR, '../../../fast_align/build/./fast_align')

#### CONSTANTS ####

#POS and NER tags

LF_ENTITY = '_entity_'
LF_ENTITY_NUM = '_entity_num_'
LF_ENTITY_TYPE = '_entity_type_'
LF_REL = '_relation_'
LF_REL_NUM = '_relation_num_'
LF_REL_DATE = '_relation_date_'
LF_REL_TIME = '_relation_time_'
LF_REL_UNARY = '_relation_unary_'
LF_REL_SUBJ = '_relation_subj_'
LF_NUM = '_number_'
LF_DATE = '_date_'
LF_TIME = '_time_'
LF_TOKS = [LF_ENTITY, LF_ENTITY_NUM, LF_ENTITY_TYPE, LF_REL, LF_REL_NUM, LF_REL_DATE, LF_REL_TIME, LF_REL_UNARY,
           LF_REL_SUBJ, LF_NUM, LF_DATE, LF_TIME]

NL_NN = '_nn_'
NL_NNP = '_nnp_'
NL_NUM = '_number_'
NL_DATE = '_date_'
NL_TIME = '_time_'
NL_ADJ = '_jj_'
NL_ENTITY = '_entity_'
NL_VERB = '_verb_'
NL_TOKS = [NL_NN, NL_NNP, NL_NUM, NL_DATE, NL_TIME, NL_ADJ, NL_ENTITY, NL_VERB]


def update(is_dev_mode_updated):
    global DATA_PATH, DATA_CROSS_PATH, ALIGNMENTS_IN_PATH, ALIGNMENTS_OUT_PATH, is_dev_mode
    is_dev_mode = is_dev_mode_updated
    DATA_PATH = DATA_DEV_MODE_PATH if is_dev_mode else DATA_TEST_MODE_PATH
    DATA_CROSS_PATH = DATA_DEV_MODE_CROSS_PATH if is_dev_mode else DATA_TEST_MODE_CROSS_PATH
    ALIGNMENTS_IN_PATH = os.path.join(ROOT_DIR, '../../../fast_align_alignments_dev/input') if is_dev_mode else os.path.join(ROOT_DIR, '../../../fast_align_alignments/input')
    ALIGNMENTS_OUT_PATH = os.path.join(ROOT_DIR, '../../../fast_align_alignments_dev/output') if is_dev_mode else os.path.join(ROOT_DIR, '../../../fast_align_alignments/output')