import math
import os
import random
import argparse

import numpy as np
import tensorflow as tf

from dataset import Dataset
from delex_data import Delexicalizer
import paths_utils

MODEL_OUT_PATH = paths_utils.MODEL_OUT_PATH


def get_ground_sim(domain, is_cross_domain, is_train):

    if is_cross_domain:
        path = DATA_OUT_CROSS_PATH
    else:
        path = DATA_OUT_PATH

    if is_train:
        suffix = '_train.tsv'
    else:
        suffix = '_test.tsv'

    # load alignment supervision
    if is_train:
        suffix_ = '_train'
    else:
        suffix_ = '_test'
    if is_cross_domain:
        domains = []
        for row in open(os.path.join(path, domain + suffix)):
            example = row.rstrip().split('\t')
            domain_ = example[6]
            if domain_+suffix_ not in domains:
                domains.append(domain_+suffix_)
    else:
        domains = [domain+suffix_]
    alignments = fast_align_utils.load_alignment(domains)


    data = []
    for row in open(os.path.join(path, domain+suffix)):
        example = row.rstrip().split('\t')
        data.append(example)
    print 'loaded {} examples'.format(len(data))

    data_matches = []
    delexicalizer = None
    for k, example in enumerate(data):
        x_delex = example[0].split(' ')
        y_delex = example[1].split(' ')
        domain = example[6]
        if delexicalizer is None or delexicalizer.domain != domain:  # initialize delexicalizer if domain has changed
            delexicalizer = Delexicalizer(domain)
        example_matches = []

        for i, y_tok_delex in enumerate(y_delex):
            if y_tok_delex in paths_utils.LF_TOKS:
                for j, x_tok_delex in enumerate(x_delex):
                    if (j, i) in alignments[k]:
                        example_matches += [(j, i, 1.0)]
                    else:
                        example_matches += [(j, i, 0.0001)]
        data_matches.append(example_matches)
    return data, data_matches


def get_tokens(docs):
    all_tokens = []
    for doc in docs:
        tokens = doc.split()
        all_tokens.append(tokens)
    return all_tokens


def get_matrices(train_raw, dev_raw, vocab_processor_nl, vocab_processor_lf):
    """
    Get matrices with word indices for each set
    :param vocab_processor: a tensorflow instance
    :return: a triplet for each set with its numpy arrays for the sentences and a list for the labels.
    """
    x_train = [example[0][0] for example in train_raw]
    y_train = [example[0][1] for example in train_raw]
    x_dev = [example[0][0] for example in dev_raw]
    y_dev = [example[0][1] for example in dev_raw]
    vocab_processor_nl.fit(x_train)
    vocab_processor_lf.fit(y_train)
    vocabulary_size_nl = len(vocab_processor_nl.vocabulary_)
    print('Total nl tokens: %d' % vocabulary_size_nl)
    vocabulary_size_lf = len(vocab_processor_lf.vocabulary_)
    print('Total lf tokens: %d' % vocabulary_size_lf)
    x_train_mat = get_matrix(x_train, vocab_processor_nl)
    y_train_mat = get_matrix(y_train, vocab_processor_lf)
    print 'finished converting train'
    x_dev_mat = get_matrix(x_dev, vocab_processor_nl)
    y_dev_mat = get_matrix(y_dev, vocab_processor_lf)
    print 'finished converting dev'

    return (x_train_mat, y_train_mat), (x_dev_mat, y_dev_mat)


def create_sims_matrix(data_raw):

    max_nl = 0
    max_lf = 0
    for example in data_raw:
        max_nl = max(max_nl, len(example[0][0].split(' ')))
        max_lf = max(max_lf, len(example[0][1].split(' ')))

    sims_mat = np.zeros([len(data_raw), max_nl, max_lf])
    for k, example in enumerate(data_raw):
        sims = example[1]
        for sim in sims:
            x_ind, y_ind, cosine = sim
            sims_mat[k, x_ind, y_ind] = cosine
    return sims_mat


def sims_to_indices(sims_mat):
    one_indices = np.where(sims_mat==1)
    indices = np.zeros([sims_mat.shape[0], sims_mat.shape[2]])
    indices[one_indices[0],one_indices[2]] = one_indices[1]
    return indices


def get_matrix(data, vocab_processor):
    """
    Convert a list of strings to a numpy array of word indices.
    """
    transform = vocab_processor.transform(data)
    return np.array(list(transform))


def BiRNN(x, n_hidden, lengths, reuse=None, name=None, input_keep_prob=1.0, output_keep_prob=1.0):
    # Forward direction cell
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, reuse=reuse)
    if input_keep_prob < 1.0 or output_keep_prob < 1.0:
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob,
                                                     variational_recurrent=True, dtype=tf.float32, input_size=x.get_shape()[-1])
    # Backward direction cell
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, reuse=reuse)
    if input_keep_prob < 1.0 or output_keep_prob < 1.0:
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob,
                                                     variational_recurrent=True, dtype=tf.float32, input_size=x.get_shape()[-1])

    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                    dtype=tf.float32, sequence_length=lengths)
    return tf.concat((outputs[0], outputs[1]), axis=2, name=name)


def train_model_ce(params, data_raw, data_raw_dev, domain):  # cross entropy loss
    vocab_processor_nl = tf.contrib.learn.preprocessing.VocabularyProcessor(
        max_document_length=params['max_document_length'],
        min_frequency=params['min_frequency'],
        tokenizer_fn=get_tokens)
    vocab_processor_lf = tf.contrib.learn.preprocessing.VocabularyProcessor(
        max_document_length=params['max_document_length'],
        min_frequency=params['min_frequency'],
        tokenizer_fn=get_tokens)

    train_raw = data_raw
    if data_raw_dev is None:
        dev_raw = data_raw[:min(100, len(data_raw))]
    else:
        dev_raw = data_raw_dev

    (x_train_mat, y_train_mat), (x_dev_mat, y_dev_mat) = get_matrices(train_raw, dev_raw, vocab_processor_nl,
                                                                      vocab_processor_lf)

    if not os.path.exists(os.path.join(MODEL_OUT_PATH, domain)):
        os.makedirs(os.path.join(MODEL_OUT_PATH, domain))
    vocab_processor_nl._tokenizer = None
    vocab_processor_lf._tokenizer = None
    vocab_processor_nl.save(os.path.join(MODEL_OUT_PATH, domain, 'vocab.nl'))
    vocab_processor_lf.save(os.path.join(MODEL_OUT_PATH, domain, 'vocab.lf'))

    train_sims = create_sims_matrix(train_raw)
    dev_sims = create_sims_matrix(dev_raw)

    dataset = Dataset(x_train_mat, y_train_mat, train_sims, params['batch_size'])

    x_dev_lengths = (x_dev_mat != 0).cumsum(1).argmax(1) + 1
    y_dev_lengths = (y_dev_mat != 0).cumsum(1).argmax(1) + 1

    x_dev_max_len = np.max(x_dev_lengths)
    y_dev_max_len = np.max(y_dev_lengths)
    dev_sims = dev_sims[:, :x_dev_max_len, :y_dev_max_len]
    x_dev_mat = x_dev_mat[:, :x_dev_max_len]
    y_dev_mat = y_dev_mat[:, :y_dev_max_len]

    embeddings_nl = tf.Variable(
        tf.random_uniform([len(vocab_processor_nl.vocabulary_), params['embedding_size']], -0.1, 0.1))
    embeddings_lf = tf.Variable(
        tf.random_uniform([len(vocab_processor_lf.vocabulary_), params['embedding_size']], -0.1, 0.1))

    # define placeholders
    x = tf.placeholder(tf.int64, [None, None], name='x')
    y = tf.placeholder(tf.int64, [None, None], name='y')
    x_lengths = tf.placeholder(tf.int64, [None], name='x_len')
    y_lengths = tf.placeholder(tf.int64, [None], name='y_len')
    similarities = tf.placeholder(tf.float32, [None, None, None], name='sims')
    targets = tf.placeholder(tf.int64, [None, None], name='targets')

    embed_x = tf.nn.embedding_lookup(embeddings_nl, x)
    embed_y = tf.nn.embedding_lookup(embeddings_lf, y)

    batch_size = tf.shape(x)[0]

    with tf.variable_scope("x"):
        bi_x = BiRNN(embed_x, params['cell_size'], x_lengths, name='rnn_x',
                     input_keep_prob=params['input_keep_prob'], output_keep_prob=params['output_keep_prob'])
    with tf.variable_scope("y"):
        bi_y = BiRNN(embed_y, params['cell_size'], y_lengths, name='rnn_y',
                     input_keep_prob=params['input_keep_prob'], output_keep_prob=params['output_keep_prob'])

    bi_y_tr = tf.transpose(bi_y, [0, 2, 1])

    W = tf.Variable(tf.truncated_normal([2 * params['cell_size'], 2 * params['cell_size']], stddev=0.1))
    b = tf.Variable(tf.zeros([1]))

    W_expand = tf.transpose(tf.tile(tf.expand_dims(W, 2), [1, 1, batch_size]), [2, 0, 1])
    bi_lin = tf.matmul(tf.matmul(bi_x, W_expand), bi_y_tr) + b
    bi_lin = tf.identity(bi_lin, name='bi_scores')

    bi_lin_t = tf.transpose(bi_lin, [0, 2, 1])  # batch x source x target
    similarities_t = tf.transpose(similarities, [0, 2, 1])

    sum_on_x = tf.reduce_sum(similarities_t, 2)  # is the sum for certain source is larger than 1, we should train on it
    seq_weights = tf.cast(tf.greater_equal(sum_on_x, tf.constant(1, dtype=tf.float32)), dtype=tf.float32)

    zero = tf.constant(0, dtype=tf.float32)
    mask_pos = tf.cast(tf.not_equal(similarities_t, zero), dtype=tf.float32)  # zero padded words we don't train on

    mask = tf.cast(tf.equal(similarities_t, zero), dtype=tf.float32) * -100  # assign low prob for padded words

    bi_lin_t_filter = (bi_lin_t * mask_pos) + mask

    cost = tf.contrib.seq2seq.sequence_loss(logits=bi_lin_t_filter, targets=targets, weights=seq_weights)

    train_step = tf.train.AdamOptimizer(params['lr']).minimize(cost, name='train_step')

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for j in range(params['num_epochs']):
            i = 0
            while dataset.has_next_batch():
                i += 1
                x_batch, y_batch, x_batch_lengths, y_batch_lengths, sims_batch = dataset.next_batch()
                target_indices = sims_to_indices(sims_batch)
                feed_dict = {x: x_batch, y: y_batch, x_lengths: x_batch_lengths,
                             y_lengths: y_batch_lengths, similarities: sims_batch, targets: target_indices}
                sess.run(train_step, feed_dict=feed_dict)

                cost_train = sess.run(cost, feed_dict=feed_dict)
                if i % 10 == 0:
                    feed_dict_dev = {x: x_dev_mat, y: y_dev_mat, x_lengths: x_dev_lengths,
                                     y_lengths: y_dev_lengths, similarities: dev_sims, targets: sims_to_indices(dev_sims)}

                    cost_dev = sess.run(cost, feed_dict=feed_dict_dev)

                    status = '{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(cost_train, cost_dev, math.sqrt(cost_train),
                                                                 math.sqrt(cost_dev), i, dataset.curr_ind, j)
                    print status

            feed_dict_dev = {x: x_dev_mat, y: y_dev_mat, x_lengths: x_dev_lengths,
                             y_lengths: y_dev_lengths, similarities: dev_sims, targets: sims_to_indices(dev_sims)}

            cost_dev = sess.run(cost, feed_dict=feed_dict_dev)
            print 'epoch={}, dev_cost={}'.format(j, math.sqrt(cost_dev))

            weights, aligns = sess.run([seq_weights, bi_lin_t_filter], feed_dict=feed_dict_dev)
            pred_targ = np.argmax(aligns, axis=2)
            gold_targ = sims_to_indices(dev_sims)
            targ = pred_targ[np.where(weights == 1)]
            gold = gold_targ[np.where(weights == 1)]
            print np.sum(targ == gold) / float(len(gold))

            saver.save(sess, os.path.join(MODEL_OUT_PATH, domain, 'cosine-model'))
            dataset.reset()


def _parse_args():
    parser = argparse.ArgumentParser(
      description='Aligner parser.',
      formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--split_dev', action='store_true',
                        help='Whether to use the dev split')
    parser.add_argument('--domains', '-d', default=[],
                          type=lambda s: [x for x in s.split(',')],
                          help=('Domains to be delexicalized represented by a comma-separated list.'))
    parser.add_argument('--cross_domain', action='store_true',
                        help='Train cross domain alignment model (train from source domains).')
    parser.add_argument('--eval_dev', action='store_true',
                        help='Evaluate on the dev set')
    parser.add_argument('--num_epochs', '-t', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--cell_size', '-c', type=int, default=150, help='Dimension of RNN cell size')
    parser.add_argument('--output_keep_prob', type=float, default=1.0, help='Output keep probability dropout rate.')
    parser.add_argument('--input_keep_prob', type=float, default=1.0, help='Input keep probability dropout rate.')

    return parser.parse_args()


def train_domain(domain, is_cross_domain, is_eval_dev, num_epochs, cell_size, output_keep_prob, input_keep_prob):
    random.seed(0)
    data, sims = get_ground_sim(domain, is_cross_domain, is_train=True)

    data_raw_dev = None
    if is_eval_dev:
        data_dev, sims_dev = get_ground_sim(domain, is_cross_domain, is_train=False)
        data_raw_dev = zip(data_dev, sims_dev)

    data_raw = zip(data, sims)

    if is_cross_domain:
        random.shuffle(data_raw)

    params = {}
    params['max_document_length'] = 100
    params['min_frequency'] = 1
    params['embedding_size'] = 100
    # params['beta'] = 0.001
    params['lr'] = 0.0002
    params['batch_size'] = 32
    params['num_epochs'] = num_epochs
    # params['hidden_size'] = 250
    params['cell_size'] = cell_size
    params['test_results'] = True
    params['output_keep_prob'] = output_keep_prob
    params['input_keep_prob'] = input_keep_prob

    tf.set_random_seed(0)

    train_model_ce(params, data_raw, data_raw_dev, domain)


class Loaded_model():

    def __init__(self, domain):

        self.vocab_processor_nl = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(os.path.join(MODEL_OUT_PATH, domain, 'vocab.nl'))
        self.vocab_processor_lf = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(os.path.join(MODEL_OUT_PATH, domain, 'vocab.lf'))
        self.vocab_processor_nl._tokenizer = get_tokens
        self.vocab_processor_lf._tokenizer = get_tokens

        self.session = tf.Session()
        saver = tf.train.import_meta_graph(os.path.join(MODEL_OUT_PATH, domain, 'cosine-model'+'.meta'))
        saver.restore(self.session, os.path.join(MODEL_OUT_PATH, domain, 'cosine-model'))

        self.graph = tf.get_default_graph()
        # extract placeholders
        self.x = self.graph.get_tensor_by_name("x:0")
        self.y = self.graph.get_tensor_by_name("y:0")
        self.x_len = self.graph.get_tensor_by_name("x_len:0")
        self.y_len = self.graph.get_tensor_by_name("y_len:0")
        self.bi_lin_score = self.graph.get_tensor_by_name("bi_scores:0")

    def classify(self, x_delexes, y_delexes, x_length):
        x_mat = self.vocab_processor_nl.transform(x_delexes)
        x_mat = np.array(list(x_mat))
        y_mat = self.vocab_processor_lf.transform(y_delexes)
        y_mat = np.array(list(y_mat))

        x_lengths = (x_mat != 0).cumsum(1).argmax(1) + 1
        y_lengths = (y_mat != 0).cumsum(1).argmax(1) + 1

        # x_max_len = np.max(x_lengths)
        x_max_len = np.max(x_length)
        y_max_len = np.max(y_lengths)

        x_mat = x_mat[:, :x_max_len]
        y_mat = y_mat[:, :y_max_len]

        feed_dict = {self.x: x_mat, self.y: y_mat, self.x_len: x_lengths, self.y_len: y_lengths}

        # extract nodes
        bi_lin_score = self.graph.get_tensor_by_name("bi_scores:0")

        bi_lin_score_pred = self.session.run(bi_lin_score, feed_dict=feed_dict)
        return bi_lin_score_pred

    def close_session(self):
        self.session.close()


if __name__ == "__main__":
    args = _parse_args()
    domains = args.domains
    is_cross_domain = args.cross_domain
    is_dev_split = args.split_dev
    is_eval_dev = args.eval_dev
    num_epochs = args.num_epochs
    cell_size = args.cell_size
    output_keep_prob = args.output_keep_prob
    input_keep_prob = args.input_keep_prob

    # set paths according to dev/test split
    paths_utils.update(is_dev_split)
    import fast_align_utils
    DATA_OUT_PATH = paths_utils.DATA_PATH
    DATA_OUT_CROSS_PATH = paths_utils.DATA_CROSS_PATH

    for domain in domains:
        train_domain(domain, is_cross_domain, is_eval_dev, num_epochs, cell_size, output_keep_prob, input_keep_prob)