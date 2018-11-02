"""Run tests on toy data for IRW models."""
import argparse
import cgi
import collections
import json
import operator
import random
import sys

import numpy
import theano
from scipy import spatial

import atislexicon
import domains
import spec as specutil
from attention import AttentionModel
from augmentation import Augmenter
from encoderdecoder import EncoderDecoderModel
from example import Example
from vocabulary import Vocabulary
from domain_stats_vocab import DomainStatsVocab
# from evaluate import Evaluator

import zero_shot.attention_utils as attention_utils
from zero_shot.delex_data import Delexicalizer
from zero_shot.inference import Predictor

MODELS = collections.OrderedDict([
    ('encoderdecoder', EncoderDecoderModel),
    ('attention', AttentionModel),
])

VOCAB_TYPES = collections.OrderedDict([
    ('raw', lambda s, e, **kwargs: Vocabulary.from_sentences(
        s, e, **kwargs)), 
    ('glove', lambda s, e, **kwargs: Vocabulary.from_sentences(
        s, e, use_glove=True, **kwargs))
])

# Global options
OPTIONS = None

# Global statistics
STATS = {}

def _parse_args():
  global OPTIONS
  parser = argparse.ArgumentParser(
      description='A neural semantic parser.',
      formatter_class=argparse.RawTextHelpFormatter
  )
  parser.add_argument('--delexicalize', action='store_true',
                      help='Run delecixalized training and inference.')
  parser.add_argument('--domain-stats', action='store_true',
                      help='Add domain statistics as a decoder input.')
  parser.add_argument('--baseline', action='store_true',
                      help='Add domain statistics as a decoder input.')
  # parser.add_argument('--epoch-eval', action='store_true',
  #                     help='Evaluate denotation accuracy after each epoch.')
  parser.add_argument('--hidden-size', '-d', type=int,
                      help='Dimension of hidden units')
  parser.add_argument('--input-embedding-dim', '-i', type=int,
                      help='Dimension of input vectors.')
  parser.add_argument('--output-embedding-dim', '-o', type=int,
                      help='Dimension of output word vectors.')
  parser.add_argument('--copy', '-p', default='none',
                      help='Way to copy words (options: [none, attention, attention-logistic]).')
  parser.add_argument('--unk-cutoff', '-u', type=int, default=0,
                      help='Treat input words with <= this many occurrences as UNK.')
  parser.add_argument('--num-epochs', '-t', default=[],
                      type=lambda s: [int(x) for x in s.split(',')], 
                      help=('Number of epochs to train (default is no training).'
                            'If comma-separated list, will run for some epochs, halve learning rate, etc.'))
  parser.add_argument('--learning-rate', '-r', type=float, default=0.1,
                      help='Initial learning rate (default = 0.1).')
  parser.add_argument('--step-rule', '-s', default='simple',
                      help='Use a special SGD step size rule (types=[simple, adagrad, rmsprop,nesterov])')
  parser.add_argument('--lambda-reg', '-l', type=float, default=0.0,
                      help='L2 regularization constant (per example).')
  parser.add_argument('--rnn-type', '-c',
                      help='type of continuous RNN model (options: [%s])' % (
                          ', '.join(specutil.RNN_TYPES)))
  parser.add_argument('--model', '-m',
                      help='type of overall model (options: [%s])' % (
                          ', '.join(MODELS)))
  parser.add_argument('--input-vocab-type',
                      help='type of input vocabulary (options: [%s])' % (
                          ', '.join(VOCAB_TYPES)), default='raw')
  parser.add_argument('--output-vocab-type',
                      help='type of output vocabulary (options: [%s])' % (
                          ', '.join(VOCAB_TYPES)), default='raw')
  parser.add_argument('--reverse-input', action='store_true',
                      help='Reverse the input sentence (intended for encoder-decoder).')
  parser.add_argument('--float32', action='store_true',
                      help='Use 32-bit floats (default is 64-bit/double precision).')
  parser.add_argument('--beam-size', '-k', type=int, default=0,
                      help='Use beam search with given beam size (default is greedy).')
  parser.add_argument('--domain', default=None,
                      help='Domain for augmentation and evaluation (options: [geoquery,atis,overnight-${domain}])')
  parser.add_argument('--use-lexicon', action='store_true',
                      help='Use a lexicon for copying (should also supply --domain)')
  parser.add_argument('--augment', '-a',
                      help=('Options for augmentation.  Format: '
                            '"nesting+entity+concat2".'))
  parser.add_argument('--aug-frac', type=float, default=0.0,
                      help='How many recombinant examples to add, relative to '
                      'training set size.')
  parser.add_argument('--distract-prob', type=float, default=0.0,
                      help='Probability to introduce distractors during training.')
  parser.add_argument('--distract-num', type=int, default=0,
                      help='Number of distracting examples to use.')
  parser.add_argument('--concat-prob', type=float, default=0.0,
                      help='Probability to concatenate examples during training.')
  parser.add_argument('--concat-num', type=int, default=1,
                      help='Number of examples to concatenate together.')
  parser.add_argument('--train-data', help='Path to training data.')
  parser.add_argument('--dev-data', help='Path to dev data.')
  parser.add_argument('--dev-frac', type=float, default=0.0,
                      help='Take this fraction of train data as dev data.')
  parser.add_argument('--dev-seed', type=int, default=0,
                      help='RNG seed for the train/dev splits (default = 0)')
  parser.add_argument('--model-seed', type=int, default=0,
                      help="RNG seed for the model's initialization and SGD ordering (default = 0)")
  parser.add_argument('--save-file', help='Path to save parameters.')
  parser.add_argument('--load-file', help='Path to load parameters, will ignore other passed arguments.')
  parser.add_argument('--stats-file', help='Path to save statistics (JSON format).')
  parser.add_argument('--shell', action='store_true', 
                      help='Start an interactive shell.')
  parser.add_argument('--server', action='store_true', 
                      help='Start an interactive web console (requires bottle).')
  parser.add_argument('--hostname', default='127.0.0.1', help='server hostname')
  parser.add_argument('--port', default=9001, type=int, help='server port')
  parser.add_argument('--theano-fast-compile', action='store_true',
                      help='Run Theano in fast compile mode.')
  parser.add_argument('--theano-profile', action='store_true',
                      help='Turn on profiling in Theano.')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  OPTIONS = parser.parse_args()
  
  # Some basic error checking
  if OPTIONS.rnn_type not in specutil.RNN_TYPES:
    print >> sys.stderr, 'Error: rnn type must be in %s' % (
        ', '.join(specutil.RNN_TYPES))
    sys.exit(1)
  if OPTIONS.model not in MODELS:
    print >> sys.stderr, 'Error: model must be in %s' % (
        ', '.join(MODELS))
    sys.exit(1)
  if OPTIONS.input_vocab_type not in VOCAB_TYPES:
    print >> sys.stderr, 'Error: input_vocab_type must be in %s' % (
        ', '.join(VOCAB_TYPES))
    sys.exit(1)
  if OPTIONS.output_vocab_type not in VOCAB_TYPES:
    print >> sys.stderr, 'Error: output_vocab_type must be in %s' % (
        ', '.join(VOCAB_TYPES))
    sys.exit(1)


def configure_theano():
  if OPTIONS.theano_fast_compile:
    theano.config.mode='FAST_COMPILE'
  else:
    theano.config.mode='FAST_RUN'
    theano.config.linker='cvm'
  if OPTIONS.theano_profile:
    theano.config.profile = True

def load_dataset(filename, domain):
  dataset = []
  with open(filename) as f:
    for line in f:
      x_delex, y_delex, x_orig, y_orig, x_orig_same, y_orig_same, src_domain, pos, src_domain_stats = line.rstrip('\n').split('\t')

      x = x_orig
      y = y_orig
      if OPTIONS.delexicalize:
        x = x_delex
        y = y_delex

      if domain:
        y = domain.preprocess_lf(y)
      dataset.append((x, y, x_orig, y_orig, x_orig_same, y_orig_same, src_domain, pos, src_domain_stats))
  return dataset

def get_input_vocabulary(dataset):
  sentences = [x[0] for x in dataset]
  constructor = VOCAB_TYPES[OPTIONS.input_vocab_type]
  if OPTIONS.float32:
    return constructor(sentences, OPTIONS.input_embedding_dim,
                       unk_cutoff=OPTIONS.unk_cutoff,
                       float_type=numpy.float32)
  else:
    return constructor(sentences, OPTIONS.input_embedding_dim,
                       unk_cutoff=OPTIONS.unk_cutoff)

def get_output_vocabulary(dataset):
  sentences = [x[1] for x in dataset]
  constructor = VOCAB_TYPES[OPTIONS.output_vocab_type]
  if OPTIONS.float32:
    return constructor(sentences, OPTIONS.output_embedding_dim,
                       float_type=numpy.float32)
  else:
    return constructor(sentences, OPTIONS.output_embedding_dim)

def update_model(model, dataset):
  """Update model for new dataset if fixed word vectors were used.
  
  Note: glove_fixed has been removed for now.
  """
  need_new_model = False
  if OPTIONS.input_vocab_type == 'glove_fixed':
    in_vocabulary = get_input_vocabulary(dataset)
    need_new_model = True
  else:
    in_vocabulary = model.in_vocabulary

  if OPTIONS.output_vocab_type == 'glove_fixed':
    out_vocabulary = get_output_vocabulary(dataset)
    need_new_model = True
  else:
    out_vocabulary = model.out_vocabulary

  if need_new_model:
    spec = model.spec
    spec.set_in_vocabulary(in_vocabulary)
    spec.set_out_vocabulary(out_vocabulary)
    model = get_model(spec)  # Create a new model!
  return model

def preprocess_data(model, raw):
  in_vocabulary = model.in_vocabulary
  out_vocabulary = model.out_vocabulary
  domain_stats_vocab = model.domain_stats_vocab
  lexicon = model.lexicon

  data = []
  for raw_ex in raw:
    x_str, y_str, x_orig, y_orig, x_orig_same, y_orig_same, src_domain, pos, src_domain_stats = raw_ex
    d_inds = [domain_stats_vocab.domain_to_index[src_domain] for x in out_vocabulary.sentence_to_indices(y_str)]
    ex = Example(x_str, y_str, x_orig, y_orig, x_orig_same, y_orig_same, src_domain, pos, in_vocabulary, out_vocabulary, d_inds, lexicon,
                 reverse_input=OPTIONS.reverse_input)
    data.append(ex)
  return data

def get_spec(in_vocabulary, out_vocabulary, lexicon, domain_stats_vocab):
  kwargs = {'rnn_type': OPTIONS.rnn_type, 'step_rule': OPTIONS.step_rule}
  if OPTIONS.copy.startswith('attention'):
    if OPTIONS.model == 'attention':
      kwargs['attention_copying'] = OPTIONS.copy
    else:
      print >> sys.stderr, "Can't use use attention-based copying without attention model"
      sys.exit(1)
  constructor = MODELS[OPTIONS.model].get_spec_class()
  return constructor(in_vocabulary, out_vocabulary, lexicon, domain_stats_vocab,
                     OPTIONS.hidden_size, **kwargs)

def get_model(spec):
  constructor = MODELS[OPTIONS.model]
  if OPTIONS.float32:
    model = constructor(spec, distract_num=OPTIONS.distract_num, float_type=numpy.float32)
  else:
    model = constructor(spec, distract_num=OPTIONS.distract_num)
  return model

def print_accuracy_metrics(name, is_correct_list, tokens_correct_list,
                           x_len_list, y_len_list, denotation_correct_list, strct_acc, infer_acc, infer_error, infer_rank):
  # Overall metrics
  num_examples = len(is_correct_list)
  num_correct = sum(is_correct_list)
  num_tokens_correct = sum(tokens_correct_list)
  num_tokens = sum(y_len_list)
  seq_accuracy = float(num_correct) / num_examples
  token_accuracy = float(num_tokens_correct) / num_tokens

  STATS[name] = {}

  # Print sequence-level accuracy
  STATS[name]['sentence'] = {
      'correct': num_correct,
      'total': num_examples,
      'accuracy': seq_accuracy,
  }
  print 'Sequence-level accuracy: %d/%d = %g' % (num_correct, num_examples, seq_accuracy)

  # Print token-level accuracy
  STATS[name]['token'] = {
      'correct': num_tokens_correct,
      'total': num_tokens,
      'accuracy': token_accuracy,
  }
  print 'Token-level accuracy: %d/%d = %g' % (num_tokens_correct, num_tokens, token_accuracy)

  # Print denotation-level accuracy
  if denotation_correct_list:
    denotation_correct = sum(denotation_correct_list)
    denotation_accuracy = float(denotation_correct)/num_examples
    STATS[name]['denotation'] = {
        'correct': denotation_correct,
        'total': num_examples,
        'accuracy': denotation_accuracy
    }
    print 'Denotation-level accuracy: %d/%d = %g' % (denotation_correct, num_examples, denotation_accuracy)

  STATS[name]['structure_correct'] = {
    'accuracy': strct_acc
  }
  print 'structure_correct = {}'.format(strct_acc)

  STATS[name]['inference_correct'] = {
    'accuracy': infer_acc
  }
  print 'inference_correct = {}'.format(infer_acc)

  STATS[name]['inference_error'] = {
    'accuracy': infer_error
  }
  print 'inference_error = {}'.format(infer_error)

  STATS[name]['inf_ave_rank'] = {
    'accuracy': infer_rank
  }
  print 'inf_ave_rank = {}'.format(infer_rank)


def find_kb_const(tok, delex_other):
  for domain in delex_other:
    for i, lex_ent in enumerate(delex_other[domain].lexicon):
      # for phrase_, const_ in lex_ent.items():
      for phrase_, const_ in delex_other[domain].lexicon[lex_ent].items():
        if const_ == tok:
          return phrase_, const_, lex_ent, domain
  return None, None, None, None


def baseline_replace(derivations, delexicalizer):
  domains = ['publications', 'restaurants', 'housing', 'recipes', 'socialnetwork', 'calendar', 'blocks']
  cands = dict()
  for l in delexicalizer.lexicon:
    cands.update(delexicalizer.lexicon[l])
    # continue
  delex_other = dict()
  for domain in domains:
    if domain != delexicalizer.domain:
      delex_other[domain] = Delexicalizer(domain)

  for deriv_ in derivations:
    out_toks = list(deriv_.y_toks)
    for i, tok in enumerate(out_toks):
      if tok == 'date' and out_toks[i-1] != 'string':
        continue
      phrase, const, group, domain_other = find_kb_const(tok, delex_other)

      if const is not None:
        pred_embed = attention_utils.get_pred_embed(phrase, delex_other[domain_other])
        if group in delexicalizer.lexicon:
          sim = attention_utils.get_similar_from_schema(delexicalizer.lexicon[group], delexicalizer, pred_embed)
        else:
          sim = attention_utils.get_similar_from_schema(cands, delexicalizer, pred_embed)
        replace = sim[0][0]
        out_toks[i] = replace
        print '{} -> {}'.format(const, replace)

    deriv_.y_toks = out_toks


def decode(model, ex, delexicalizer, loaded_model, predictor):
  if OPTIONS.beam_size == 0:
    derivations = model.decode_greedy(ex, max_len=100)
  else:
    derivations = model.decode_beam(ex, beam_size=OPTIONS.beam_size)

    if not OPTIONS.delexicalize:
      if OPTIONS.baseline:
        baseline_replace(derivations, delexicalizer)
      for deriv_ in derivations:
        deriv_.y_toks_delex = deriv_.y_toks
      return derivations

    # aggregate derivations
    x_delexes = []
    y_delexes = []
    for j, deriv in enumerate(derivations):
      x_delexes.append(ex.x_str)
      y_delexes.append(' '.join(deriv.y_toks))
    x_len = len(x_delexes[0].split(' '))  # calc x length. We do it for the string in case there is an uknown word at the end
    attention_learned = loaded_model.classify(x_delexes, y_delexes, x_len)

    # do lexicalizing here
    check_next_deriv = True
    for j, deriv in enumerate(derivations):
      if j > 10:
        break
      if not check_next_deriv:
        break
      y_toks_delex = []
      y_toks = deriv.y_toks

      print 'derivation {}'.format(j)

      # alignments = numpy.array(deriv.attention_list)[:-1, :-1].T
      alignments = attention_learned[j, :, :][:len(ex.x_toks)]

      x_delex_toks_to_run = ex.x_toks
      if predictor.domain == 'calendar' and '_date_' in deriv.y_toks:  # switch calender _date_ abstract tokens to _time_
        x_delex_time_toks, y_delex_time_toks = predictor.preprocess_time_2(ex.x_toks, deriv.y_toks, ex.pos, alignments)
        if x_delex_time_toks is not None and y_delex_time_toks is not None:
          x_delex_toks_to_run = x_delex_time_toks
          deriv.y_toks = y_delex_time_toks

      y_pred, is_executed, succ_num = predictor.score_derivation(ex.x_orig_same.split(' '), x_delex_toks_to_run, deriv.y_toks, alignments)
      check_next_deriv = not is_executed  # if executed successfully, no point checking next derivations
      if not check_next_deriv:
        ex.x_toks = x_delex_toks_to_run

      y_toks_delex = y_pred.split(' ')

      # if deriv.y_toks == deriv.example.y_toks:  # structure is correct
      #   if y_toks_delex != deriv.example.y_orig.split():  # inference is correct
      #     print 'bad inference! (gold lf printed first)'
      #     print ' '.join(deriv.y_toks)
      #     print deriv.example.y_orig
      #     print ' '.join(y_toks_delex)

      deriv.y_toks = y_toks_delex  # switch tokens
      deriv.y_toks_delex = y_toks  # the delexicalized version
      deriv.infer_succ = succ_num

  return derivations

def get_similar_from_schema(candidates, delexicalizer, x_token_emb):
  dists = {}
  for cand in candidates:
    embeds = []
    for token in cand:
      if token in delexicalizer.embeddings:
        embeds.append(delexicalizer.get_embedding(token))
    ent_embed = numpy.mean(numpy.array(embeds), axis=0)
    # dist = numpy.linalg.norm(x_token_emb - ent_embed)
    dist = spatial.distance.cosine(x_token_emb, ent_embed)
    dists[candidates[cand]] = dist
  dists_sorted = sorted(dists.items(), key=operator.itemgetter(1))
  return dists_sorted[0][0]

def evaluate(name, model, dataset, domain=None):
  """Evaluate the model. """
  in_vocabulary = model.in_vocabulary
  out_vocabulary = model.out_vocabulary

  delexicalizer = Delexicalizer(domain.subdomain)

  is_correct_list = []
  tokens_correct_list = []
  x_len_list = []
  y_len_list = []

  if domain:
    # all_derivs = [decode(model, ex, delexicalizer) for ex in dataset]

    if OPTIONS.delexicalize:
      loaded_model = attention_utils.get_trained_model(domain.subdomain)
      predictor = Predictor(domain.subdomain)
    else:
      loaded_model = None
      predictor = None
    all_derivs = []
    for i, ex in enumerate(dataset):
      print 'started inference for example {}'.format(i)
      all_derivs.append(decode(model, ex, delexicalizer, loaded_model, predictor))

    # true_answers = [ex.y_str for ex in dataset]
    true_answers = [ex.y_orig for ex in dataset]

    derivs, denotation_correct_list = domain.compare_answers(true_answers, all_derivs)

    true_answers_delex = [ex.y_str for ex in dataset]

    struct_corr = 0
    infer_corr = 0
    total_infered = 0
    error_infered = 0
    infer_succ_num = 0
    if OPTIONS.delexicalize:
      for derivs_ in all_derivs:
        for deriv_ in derivs_:
          if deriv_.y_toks_delex is not None:
            total_infered += 1
            succ_num = deriv_.infer_succ
            if succ_num is None:
              error_infered += 1
            else:
              infer_succ_num += succ_num
            if deriv_.y_toks_delex == deriv_.example.y_toks:  # structure is correct
              struct_corr += 1
              if deriv_.y_toks == deriv_.example.y_orig.split(): # inference is correct
                infer_corr += 1

    strct_acc = struct_corr / float(len(true_answers_delex))
    if struct_corr == 0:
      infer_acc = 0.0
    else:
      infer_acc = infer_corr / float(struct_corr)

    if total_infered == 0:
      infer_error = 0
      infer_rank = 0
    else:
      infer_error = error_infered / float(total_infered)
      infer_rank = infer_succ_num / float(total_infered - error_infered)

  else:
    derivs = [decode(model, ex)[0] for ex in dataset]
    denotation_correct_list = None

  for i, ex in enumerate(dataset):

    if derivs[i] is None:
      continue

    y_pred_toks_delex = derivs[i].y_toks_delex
    if y_pred_toks_delex is None:
      y_pred_toks_delex_str = '_'
    else:
      y_pred_toks_delex_str = ' '.join(y_pred_toks_delex)
    print 'Example %d' % i
    print '  x_orig = "%s"' % ex.x_orig
    print '  x      = "%s"' % ex.x_str
    print '  y      = "%s"' % ex.y_str
    print '  y_p_de = "%s"' % y_pred_toks_delex_str
    print '  y_orig = "%s"' % ex.y_orig
    prob = derivs[i].p
    y_pred_toks = derivs[i].y_toks
    y_pred_str = ' '.join(y_pred_toks)

    # Compute accuracy metrics
    is_correct = (y_pred_str == ex.y_orig)
    tokens_correct = sum(a == b for a, b in zip(y_pred_toks, ex.y_orig.split(' ')))
    is_correct_list.append(is_correct)
    tokens_correct_list.append(tokens_correct)
    x_len_list.append(len(ex.x_toks))
    y_len_list.append(len(ex.y_toks))
    print '  y_pred = "%s"' % y_pred_str
    print '  sequence correct = %s' % is_correct
    print '  token accuracy = %d/%d = %g' % (
        tokens_correct, len(ex.y_toks), float(tokens_correct) / len(ex.y_toks))
    if denotation_correct_list:
      denotation_correct = denotation_correct_list[i]
      print '  denotation correct = %s' % denotation_correct
  print_accuracy_metrics(name, is_correct_list, tokens_correct_list,
                         x_len_list, y_len_list, denotation_correct_list, strct_acc, infer_acc, infer_error, infer_rank)

def run_shell(model):
  print '==== Neural Network Semantic Parsing REPL ===='
  print ''
  print 'Enter an utterance:'
  while True:
    s = raw_input('> ').strip()
    example = Example(s, '', model.in_vocabulary, model.out_vocabulary,
                      model.lexicon, reverse_input=OPTIONS.reverse_input)
    print ''
    print 'Result:'
    preds = decode(model, example)
    for prob, y_toks in preds[:10]:
      y_str = ' '.join(y_toks)
      print '  [p=%f] %s' % (prob, y_str)
    print ''

def make_heatmap(x_str, y_str, attention_list, copy_list):
  """Make an HTML heatmap of attention."""
  def css_color(r, g, b):
    """r, g, b are in 0-1, make """
    r2 = int(r * 255)
    g2 = int(g * 255)
    b2 = int(b * 255)
    return 'rgb(%d,%d,%d)' % (r2, g2, b2)

  x_toks = [cgi.escape(w) for w in x_str.split(' ')] + ['EOS']
  if y_str == '':
    y_toks = ['EOS']
  else:
    y_toks = [cgi.escape(w) for w in y_str.split(' ')] + ['EOS']
  lines = ['<table>', '<tr>', '<td/>']
  for w in y_toks:
    lines.append('<td>%s</td>' % w)
  lines.append('</tr>')
  for i, w in enumerate(x_toks):
    lines.append('<tr>')
    lines.append('<td>%s</td>' % w)
    for j in range(len(y_toks)):
      do_copy = copy_list[j]
      if do_copy:
        color = css_color(1 - attention_list[j][i], 1 - attention_list[j][i], 1)
      else:
        color = css_color(1, 1 - attention_list[j][i], 1 - attention_list[j][i])
      lines.append('<td/ style="background-color: %s">' % color)
    lines.append('</tr>')
  lines.append('</table>')
  return '\n'.join(lines)

def run_server(model, hostname='127.0.0.1', port=9001):
  import bottle
  print '==== Neural Network Semantic Parsing Server ===='

  app = bottle.Bottle()
  
  @app.route('/debug')
  def debug():
    content = make_heatmap(
        'what states border texas',
        'answer ( A , ( state ( A ) , next_to ( A , B ) , const ( B , stateid ( texas ) ) ) )',
        [[0.0, 0.25, 0.5, 0.75, 1.0]] * 29)
    return bottle.template('main', prompt='Enter a new query', content=content)

  @app.route('/post_query')
  def post_query():
    query = bottle.request.params.get('query')
    print 'Received query: "%s"' % query
    example = Example(query, '', model.in_vocabulary, model.out_vocabulary,
                      model.lexicon, reverse_input=OPTIONS.reverse_input)
    preds = decode(model, example)
    lines = ['<b>Query: "%s"</b>' % query, '<ul>']
    for i, deriv in enumerate(preds[:10]):
      y_str = ' '.join(deriv.y_toks)
      lines.append('<li> %d. [p=%f] %s' % (i, deriv.p, y_str))
      lines.append(make_heatmap(query, y_str, deriv.attention_list, deriv.copy_list))
    lines.append('</ul>')

    content = '\n'.join(lines)
    return bottle.template('main', prompt='Enter a new query', content=content)

  @app.route('/')
  def index():
    return bottle.template('main', prompt='Enter a query', content='')

  bottle.run(app, host=hostname, port=port)

def load_raw_all(domain=None):
  # Load train, and dev too if dev-frac was provided
  random.seed(OPTIONS.dev_seed)
  if OPTIONS.train_data:
    train_raw = load_dataset(OPTIONS.train_data, domain=domain)
    if OPTIONS.dev_frac > 0.0:
      num_dev = int(round(len(train_raw) * OPTIONS.dev_frac))
      random.shuffle(train_raw)
      dev_raw = train_raw[:num_dev]
      train_raw = train_raw[num_dev:]
      print >> sys.stderr, 'Split dataset into %d train, %d dev examples' % (
          len(train_raw), len(dev_raw))
    else:
      dev_raw = None
  else:
    train_raw = None
    dev_raw = None

  # Load dev data from separate file
  if OPTIONS.dev_data:
    if dev_raw:
      # Overwrite dev frac from before, if it existed
      print >> sys.stderr, 'WARNING: Replacing dev-frac dev data with dev-data'
    dev_raw = load_dataset(OPTIONS.dev_data, domain=domain)

  return train_raw, dev_raw

def get_augmenter(train_raw, domain):
  if OPTIONS.augment:
    aug_types = OPTIONS.augment.split('+')
    augmenter = Augmenter(domain, train_raw, aug_types)
    return augmenter
  else:
    return None


def get_lexicon():
  if OPTIONS.use_lexicon:
    if OPTIONS.domain == 'atis':
      return atislexicon.get_lexicon()
    raise Exception('No lexicon for domain %s' % OPTIONS.domain)
  return None

def init_spec(train_raw, dev_raw):
  if OPTIONS.load_file:
    print >> sys.stderr, 'Loading saved params from %s' % OPTIONS.load_file
    spec = specutil.load(OPTIONS.load_file)
  elif OPTIONS.train_data:
    print >> sys.stderr, 'Initializing parameters...'
    in_vocabulary = get_input_vocabulary(train_raw)
    out_vocabulary = get_output_vocabulary(train_raw)
    domain_stats_vocab = DomainStatsVocab(train_raw+dev_raw, is_use=OPTIONS.domain_stats)
    lexicon = get_lexicon()
    spec = get_spec(in_vocabulary, out_vocabulary, lexicon, domain_stats_vocab)
  else:
    raise Exception('Must either provide parameters to load or training data.')
  return spec

def evaluate_train(model, train_data, domain=None):
  print >> sys.stderr, 'Evaluating on training data...'
  print 'Training data:'
  evaluate('train', model, train_data, domain=domain)

def evaluate_dev(model, dev_raw, domain=None):
  print >> sys.stderr, 'Evaluating on dev data...'
  dev_model = update_model(model, dev_raw)
  dev_data = preprocess_data(dev_model, dev_raw)
  print 'Dev data:'
  evaluate('dev', dev_model, dev_data, domain=domain)

def write_stats():
  if OPTIONS.stats_file:
    out = open(OPTIONS.stats_file, 'w')
    print >>out, json.dumps(STATS)
    out.close()

def run():
  configure_theano()
  domain = None
  if OPTIONS.domain:
    domain = domains.new(OPTIONS.domain)
  train_raw, dev_raw = load_raw_all(domain=domain)
  random.seed(OPTIONS.model_seed)
  numpy.random.seed(OPTIONS.model_seed)
  spec = init_spec(train_raw, dev_raw)
  model = get_model(spec)

  if train_raw:
    train_data = preprocess_data(model, train_raw)
    random.seed(OPTIONS.model_seed)
    dev_data = None
    if dev_raw:
      dev_data = preprocess_data(model, dev_raw)
    augmenter = get_augmenter(train_raw, domain)
    if not OPTIONS.load_file:
      evaluator = None
      # if OPTIONS.epoch_eval:
      #   evaluator = Evaluator(domain, OPTIONS)
      model.train(train_data, T=OPTIONS.num_epochs, eta=OPTIONS.learning_rate,
                  dev_data=dev_data, l2_reg=OPTIONS.lambda_reg,
                  distract_prob=OPTIONS.distract_prob,
                  distract_num=OPTIONS.distract_num,
                  concat_prob=OPTIONS.concat_prob, concat_num=OPTIONS.concat_num,
                  augmenter=augmenter, aug_frac=OPTIONS.aug_frac, evaluator=evaluator)

  if OPTIONS.save_file:
    print >> sys.stderr, 'Saving parameters...'
    spec.save(OPTIONS.save_file)

  # if train_raw:
  #   evaluate_train(model, train_data, domain=domain)
  if dev_raw:
    evaluate_dev(model, dev_raw, domain=domain)

  write_stats()

  if OPTIONS.shell:
    run_shell(model)
  elif OPTIONS.server:
    run_server(model, hostname=OPTIONS.hostname, port=OPTIONS.port)

def main():
  _parse_args()
  print OPTIONS
  print >> sys.stderr, OPTIONS
  run()

if __name__ == '__main__':
  main()
