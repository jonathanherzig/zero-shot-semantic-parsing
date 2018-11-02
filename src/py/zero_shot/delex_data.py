import os
import sys
import numpy as np
import time
import editdistance
import itertools
import re
from itertools import product
from collections import Counter

import paths_utils
import enchant

LEX_PATH = paths_utils.LEX_PATH
DATA_IN_PATH = paths_utils.DATA_RAW_PATH
DATA_OUT_PATH = paths_utils.DATA_TEST_MODE_PATH
EMBEDDING_PATH = paths_utils.EMBEDDING_PATH
NLP_PATH = paths_utils.NLP_PATH


class Delexicalizer(object):

    def __init__(self, domain):

        self.domain = domain
        self.is_train = True

        self.nlp = None
        self.embeddings = None
        self.lexicon = self.load_lexicon(LEX_PATH, domain)
        self.domain_stats = self.get_domain_stats()

        self.excluded_verbs = ['did', 'do', 'does', 'is', 'are', 'be', 'been', 'has', 'have', 'had', 'was', 'were', 'will', 'show', 'find', 'name', 'list']
        self.excluded_nouns = ['number', 'average', 'total', 'largest']

        self.cands = dict()
        self.cands['_entity_'] = set(['_entity_'])
        self.cands['_entity_type_'] = set(['_nn_', '_nns_', '_nnp_', '_nnps_', '_verb_', '_jj_'])
        self.cands['_entity_num_'] = set(['_nn_', '_nns_', '_nnp_', '_nnps_', '_verb_', '_jj_'])
        self.cands['_relation_'] = set(['_nn_', '_nns_', '_nnp_', '_nnps_', '_verb_', '_jj_'])
        self.cands['_relation_num_'] = set(['_nn_', '_nns_', '_nnp_', '_nnps_', '_verb_', '_jj_'])
        self.cands['_relation_date_'] = set(['_nn_', '_nns_', '_nnp_', '_nnps_', '_verb_', '_jj_'])
        self.cands['_relation_time_'] = set(['_nn_', '_nns_', '_nnp_', '_nnps_', '_verb_', '_jj_'])
        self.cands['_relation_unary_'] = set(['_nn_', '_nns_', '_nnp_', '_nnps_', '_verb_', '_jj_'])
        self.cands['_relation_subj_'] = set(['_nn_', '_nns_', '_nnp_', '_nnps_', '_verb_', '_jj_'])
        self.cands['_number_'] = set(['_number_'])
        self.cands['_date_'] = set(['_date_'])
        self.cands['_time_'] = set(['_time_'])

        self.word2num = { "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

    def get_domain_stats(self, ):
        #ignore  self.relation_subj , self.relation_time
        schema = [paths_utils.LF_ENTITY, paths_utils.LF_ENTITY_NUM, paths_utils.LF_ENTITY_TYPE, paths_utils.LF_REL, paths_utils.LF_REL_NUM, paths_utils.LF_REL_DATE,
                  paths_utils.LF_REL_UNARY]
        stats = []
        count = 0
        for i, item in enumerate(schema):
            if item in self.lexicon:
                count += len(self.lexicon[item])
        for i, item in enumerate(schema):
            if item in self.lexicon:
                stats.append(str(len(self.lexicon[item])/float(count)))
            else:
                stats.append(str(0.0))
        return ','.join(stats)

    def load_lexicon(self, lex_path, domain):
        file = os.path.join(lex_path, domain + '.grammar.proc')
        lexicon = {}
        for row in open(file):
            parts = row.strip().split('\t')
            name = parts[0]
            symbol = parts[1]
            type = parts[2]

            if type not in lexicon:
                lexicon[type] = dict()

            name_tup = tuple(name.split(' '))
            if name_tup not in lexicon[type]:
                lexicon[type][tuple(name.split(' '))] = symbol
        return lexicon

    def delex_token(self, p, is_keep_time=False):
        pos = p[1]
        text = p[0]

        if self.domain=='blocks' and pos == 'IN' and text.lower() in ['above', 'below']:
            return paths_utils.NL_NN

        if pos in paths_utils.NL_TOKS:
            if pos == paths_utils.NL_VERB and text.lower() in self.excluded_verbs:
                return text.lower()
            elif (pos == paths_utils.NL_NN or pos == paths_utils.NL_NNP) and text.lower() in self.excluded_nouns:
                return text.lower()
            elif pos == paths_utils.NL_ADJ:
                return text.lower()
            elif pos == paths_utils.NL_TIME and not is_keep_time:
                return paths_utils.NL_DATE
            else:
                return pos
        else:
            return text.lower()

    def split_compound(self, word):
        if word.isdigit():
            return None
        language = 'en_us'
        dictionary = enchant.Dict(language)
        max_index = len(word)
        for index, char in enumerate(word):
            left_compound = word[0:max_index - index]
            right_compound = word[max_index - index:max_index]
            if index > 0 and len(left_compound) > 1 and len(right_compound) > 1 and dictionary.check(left_compound) and dictionary.check(right_compound):
                return (left_compound, right_compound)
        return None

    def delex_domain(self, domain, is_train):
        """
        Delexicalize data for some domain
        :param is_train: indicates train or test set
        :return:
        """
        dataset = '_train' if is_train else '_test'
        file = os.path.join(DATA_IN_PATH, domain + dataset + '.tsv')
        if not os.path.exists(file):
            return

        self.load_nlp()

        writer = open(os.path.join(DATA_OUT_PATH, domain + dataset + '.tsv'), 'wb')

        num_spans = 0
        num_ents = 0
        for l in open(file):
            x, y = l.rstrip('\n').split('\t')
            x = ' '.join(x.split())  # remove multiple spaces
            y = ' '.join(y.split())

            x_delex, y_delex, x_same, y_same, tags, num_spans_, num_ents_ = self.delex_train_instance(x, y)
            num_ents += num_ents_
            num_spans += num_spans_

            if '_relation_subj_' in y_delex.split(): # ignore neo-davidsonian semantics
                continue

            writer.write(x_delex + '\t' + y_delex + '\t' + x + '\t' + y + '\t' + x_same + '\t' + y_same + '\t' + self.domain + '\t' + tags + '\t' + self.domain_stats + '\n')
        writer.close()

        # print num_ents
        # print num_spans

    def split_num_words(self, x_toks):
        x_toks_edit = []
        for tok in x_toks:
            regex = re.compile(r'(\d{1,})([a-z]+)')
            tok_ = regex.sub(r'\1 \2', tok)
            if tok != tok_:
                if regex.sub(r'\2', tok) not in ['st', 'nd', 'rd', 'th']:  # don't split date tokens
                    x_toks_edit += tok_.split(' ')
                    continue
            x_toks_edit.append(tok)
        return x_toks_edit

    def delex_date_time(self, pos, ner):
        for i, pair in enumerate(ner):
            if pair[1] == paths_utils.NL_DATE:
                token = pos[i][0].lower()
                cands = [[str(k) for k in range(2001, 2011)],
                         [str(k) for k in range(32)],
                         ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
                          'september', 'october', 'november', 'december',
                          'jan', 'feb', 'mar', 'apr', 'aug', 'sept', 'oct', 'nov', 'dec'],
                         ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th', '11th',
                          '12th', '13th', '14th', '15th', '16th', '17th', '18th', '19th', '20th', '21st',
                          '22nd', '23rd', '24th', '25th', '26th', '27th', '28th', '29th', '30th', '31st']]
                for cand in cands:
                    if token in cand:
                        pos[i] = pair
                        break
            elif pair[1] == paths_utils.NL_TIME:
                token = pos[i][0]
                hour = [str(k) for k in range(1, 13)]
                time = ['am', 'pm']
                cross = [item[0] + item[1] for item in list(product(*[hour, time]))]
                if token.lower() in hour or token.lower() in time or token.lower() in cross:
                    pos[i] = pair

        self.del_consecutive(pos, paths_utils.NL_DATE)
        self.del_consecutive(pos, paths_utils.NL_TIME)
        return pos

    def del_consecutive(self, pos, tag):
        ind_to_remove = []
        for i, p in enumerate(pos):  # merge consecutive times
            text = p[0]
            if i > 0:
                if p[1] == tag and pos[i - 1][1] == tag:
                    text_prev = pos[i - 1][0]
                    pos[i] = (text_prev + '_' + text, p[1])
                    ind_to_remove.append(i - 1)
        for ind in sorted(ind_to_remove, reverse=True):
            del pos[ind]

    def split_compounds(self, x_toks):
        x_toks_edit = []
        for tok in x_toks:
            emb = self.get_embedding(tok)
            if emb is None:
                parts = self.split_compound(tok)
                if parts is None:
                    x_toks_edit.append(tok)
                else:
                    x_toks_edit += parts
            else:
                x_toks_edit.append(tok)
        return x_toks_edit

    def process_nlp(self, x):
        truecase = self.truecase(x, self.nlp)
        x_truecase = ' '.join([w[1] for w in truecase])
        ner = self.nlp.ner(x_truecase)
        pos = self.pos_joint(x_truecase, self.nlp)
        for i, p in enumerate(pos):  # fix parsing bug of two consecutive numbers
            text = p[0]
            if i > 0 and p[1] == 'CD' and pos[i - 1][1] == 'CD':
                fixed_pos = self.pos_joint(self.truecase(text, self.nlp)[0][1], self.nlp)[0][1]
                if fixed_pos != 'CD':
                    pos[i] = (text, fixed_pos)
        return pos, ner

    def map_to_special_tags(self, input):
        tag_to_sepc = {}
        tag_to_sepc['NN'] = paths_utils.NL_NN
        tag_to_sepc['NNS'] = paths_utils.NL_NN
        tag_to_sepc['NNP'] = paths_utils.NL_NNP
        tag_to_sepc['NNPS'] = paths_utils.NL_NNP
        tag_to_sepc['CD'] = paths_utils.NL_NUM
        tag_to_sepc['DATE'] = paths_utils.NL_DATE
        tag_to_sepc['TIME'] = paths_utils.NL_TIME
        tag_to_sepc['JJ'] = paths_utils.NL_ADJ
        tag_to_sepc['_entity_'] = paths_utils.NL_ENTITY
        tag_to_sepc['VB'] = paths_utils.NL_VERB
        tag_to_sepc['VBD'] = paths_utils.NL_VERB
        tag_to_sepc['VBG'] = paths_utils.NL_VERB
        tag_to_sepc['VBN'] = paths_utils.NL_VERB
        tag_to_sepc['VBP'] = paths_utils.NL_VERB
        tag_to_sepc['VBZ'] = paths_utils.NL_VERB

        for i, tup in enumerate(input):
            tag = tup[1]
            if tag in tag_to_sepc:
                input[i] = (tup[0], tag_to_sepc[tag])

        return input

    def find_type(self, y_tok):
        for type in self.lexicon:
            if y_tok in self.lexicon[type].values():
                return type
        return None

    def delex_train_instance(self, x, y):
        x_toks = x.split()
        x_toks = self.split_num_words(x_toks)
        x_toks = self.split_compounds(x_toks)

        pos, ner = self.process_nlp(' '.join(x_toks))
        pos = self.map_to_special_tags(pos)
        ner = self.map_to_special_tags(ner)
        tags = self.delex_date_time(pos, ner)

        x_same = ' '.join([p[0].lower() for p in tags])
        x_delex = ' '.join([self.delex_token(p) for p in tags])
        tags = ' '.join([p[1] for p in tags])

        # version of lexicalized tokens the same length of the delexicalized version
        y_same = str(y)
        y_rep = re.sub(r'\( date ([\d|-]+) ([\d|-]+) ([\d|-]+) \)', r'( date \1_\2_\3 )', y_same)
        y_rep = re.sub(r'\( time ([\d|-]+) ([\d|-]+) \)', r'( time \1_\2 )', y_rep)
        y_toks_same = y_rep.split()

        # handle logical form
        y_rep = re.sub(r'\( date ([\d|-]+) ([\d|-]+) ([\d|-]+) \)', r'( date _date_ )', y)
        y_rep = re.sub(r'\( time ([\d|-]+) ([\d|-]+) \)', r'( date _date_ )', y_rep)  # treat time as date

        y_toks = y_rep.split()

        y_toks_delex = []
        for i, y_tok in enumerate(y_toks):
            if i >= 1 and y_toks[i - 1] == 'number':
                y_toks_delex.append('_number_')
                continue
            elif i >= 1 and y_tok == 'date' and y_toks[i - 1] == '(': # date can be also a relation
                y_toks_delex.append('date')
                continue
            else:
                type = self.find_type(y_tok)
                if type is not None:
                    y_toks_delex.append(type)
                    continue
            y_toks_delex.append(y_tok)
        y_delex = ' '.join(y_toks_delex)
        y_same = ' '.join(y_toks_same)

        if self.is_train:
            num_spans_, num_ents_, x_delex, x_same, tags = self.replace_entities(x_delex.split(), x_same.split(), y_delex.split(), y_same.split(), tags.split())
        else:
            num_spans_, num_ents_, x_delex, x_same, tags = self.replace_entities_test(x_delex.split(), x_same.split(), tags.split())

        return x_delex, y_delex, x_same, y_same, tags, num_spans_, num_ents_


    def load_nlp(self):
        from stanfordcorenlp import StanfordCoreNLP
        if self.nlp is None:
            self.nlp = StanfordCoreNLP(NLP_PATH, memory='8g')

    def truecase(self, sentence, nlp):
        r_dict = nlp._request('truecase', sentence)
        words = []
        truecases = []
        for s in r_dict['sentences']:
            for token in s['tokens']:
                words.append(token['word'])
                truecases.append(token['truecaseText'])
        return list(zip(words, truecases))

    def pos_joint(self, sentence, nlp):
        r_dict = nlp._request('parse', sentence)
        words = []
        tags = []
        for s in r_dict['sentences']:
            for token in s['tokens']:
                words.append(token['word'])
                tags.append(token['pos'])
        return list(zip(words, tags))

    def get_embedding(self, word):
        # load embeddings here
        if self.embeddings is not None:
            return self.embeddings.get(word)

        print 'loading embeddings from file'
        embeddings = {}
        for l in open(EMBEDDING_PATH, 'rb'):
            parts = l.split(' ')
            curr_word = parts[0]
            embeddings[curr_word] = np.array(parts[1:], dtype=np.float32)
        self.embeddings = embeddings
        return self.embeddings.get(word)

    def replace_entities(self, x_delex, x_same, y_delex, y_same, tags):

        replaced_entites = set()

        x_delex_rep = x_delex
        x_same_rep = x_same
        tags_rep = tags
        num_ents = 0
        num_spans = 0
        for i, y__delex_tok in enumerate(y_delex):
            if y__delex_tok != '_entity_':
                continue
            entity_names = []
            y_tok = y_same[i]
            if y_tok in replaced_entites:
                continue
            replaced_entites.add(y_tok)
            num_ents += 1
            for entity_name, v in self.lexicon[paths_utils.LF_ENTITY].items():
                if y_tok == v:
                    entity_names.append(entity_name)
                    span = self.find_entity_spans(x_same_rep, entity_name)
                    if span is not None:
                        x_delex_rep = x_delex_rep[:span[0]] + ['_entity_'] + x_delex_rep[span[1]:]
                        x_same_rep = x_same_rep[:span[0]] + ['_'.join(x_same_rep[span[0]: span[1]])] + x_same_rep[span[1]:]
                        tags_rep = tags_rep[:span[0]] + ['_'.join(tags_rep[span[0]: span[1]])] + tags_rep[span[1]:]
                        if len(x_delex_rep)!= len(x_same_rep):
                            print 'here'
                            sys.exit(1)
                        num_spans += 1
        return num_spans, num_ents, ' '.join(x_delex_rep), ' '.join(x_same_rep), ' '.join(tags_rep)

    def replace_entities_test(self, x_delex, x_same, tags):
        x_delex_rep = x_delex
        x_same_rep = x_same
        tags_rep = tags
        num_ents = 0
        num_spans = 0
        entity_names = []
        for entity_name, v in self.lexicon[paths_utils.LF_ENTITY].items():
            entity_names.append(entity_name)
            span = self.find_entity_spans(x_same_rep, entity_name)
            if span is not None:
                x_delex_rep = x_delex_rep[:span[0]] + ['_entity_'] + x_delex_rep[span[1]:]
                x_same_rep = x_same_rep[:span[0]] + ['_'.join(x_same_rep[span[0]: span[1]])] + x_same_rep[span[1]:]
                tags_rep = tags_rep[:span[0]] + ['_'.join(tags_rep[span[0]: span[1]])] + tags_rep[span[1]:]
                if len(x_delex_rep)!= len(x_same_rep):
                    print 'here'
                    sys.exit(1)
                num_spans += 1
        return num_spans, num_ents, ' '.join(x_delex_rep), ' '.join(x_same_rep), ' '.join(tags_rep)

    def find_entity_spans(self, tokens, entity):
        span = self.find_span(tokens, entity, entity)
        if span is not None:
            return span
        for i in range(len(entity) - 1, 0, -1):
            sub_entities = list(itertools.combinations(entity, i))
            sub_entities_filtered = [sub_entity for sub_entity in sub_entities if
                                     sub_entity[0] == entity[0] or sub_entity[-1] == entity[-1]]

            for sub_entity in sub_entities_filtered:
                span = self.find_span(tokens, sub_entity, entity)
                if span is not None:
                    return span
        return span

    def find_span(self, tokens, entity, full_entity):
        short_token_len = 3
        span = None
        max_len = len(entity)
        cand_spans = []
        if max_len == 1 and len(entity[0]) < short_token_len:  # skip searching single token entities, when the token is short
            return None
        for group in self.lexicon:
            if entity in self.lexicon[group] and entity != full_entity:  # if sub-entity is contained in some other special nl, ignore
                return None
        for i in range(len(tokens) - max_len + 1):  # search if entity fully appears in text
            d = 0
            for text_token, entity_token in zip(tokens[i:(i + max_len)], entity):
                if len(text_token) < short_token_len:  # for short token, only allow exact match
                    if text_token != entity_token:
                        d += 2
                else:
                    d_local = editdistance.eval(text_token, entity_token)
                    if self.is_train:
                        d += d_local
                    else:  # in test time be more restrictive. Only allow plural.
                        if d_local == 0 or (d_local == 1 and text_token[-1] == 's' and text_token[:-1] == entity_token):
                            d += d_local
                        else:
                            d += 2
            if d <= 1:
                span = (i, i + max_len)
                cand_spans.append((d, span))
        if len(cand_spans) > 0:
            cand_spans.sort(key=lambda tup: tup[0])
            return cand_spans[0][1]
        return span


def globaly_delex_adj(domains):
    counters = dict()
    for domain in domains:
        input_file = os.path.join(DATA_OUT_PATH, domain + '_train.tsv')
        if os.path.exists(input_file):
            counters[domain] = get_domain_adj_stats(input_file)
    for domain in domains:
        input_file = os.path.join(DATA_OUT_PATH, domain + '_train.tsv')
        if os.path.exists(input_file):
            replace_adjs(input_file, domain, counters)
        input_file = os.path.join(DATA_OUT_PATH, domain + '_test.tsv')
        if os.path.exists(input_file):
            replace_adjs(input_file, domain, counters)


def get_domain_adj_stats(input_file):
    counter = Counter()
    for row in open(input_file, 'rb'):   # read training data
        example = row.rstrip().split('\t')
        x = example[4].split()
        tags = example[7].split()
        for i, tag in enumerate(tags):
            if tag == '_jj_':
                counter.update([x[i]])
    return counter


def replace_adjs(file, domain, counters):
    out_domain = set()
    for d in counters:
        if d != domain:
            out_domain |= set(counters[d].keys())

    examples = []
    for row in open(file, 'rb'):  # read training data
        examples.append(row.rstrip().split('\t'))

    with open(file, 'wb') as f:
        for example in examples:
            x = example[4].split()
            x_delex = example[0].split()
            tags = example[7].split()
            for i, tag in enumerate(tags):
                if tag == '_jj_' and x[i] not in out_domain and x_delex[i] != '_entity_':
                    x_delex[i] = '_jj_'
            example[0] = ' '.join(x_delex)
            f.write('\t'.join(example)+'\n')


def delex_domains(domains):
    for domain in domains:
        print 'delexicalizing {}'.format(domain)
        delexicalizer = Delexicalizer(domain)

        delexicalizer.is_train = True
        delexicalizer.delex_domain(domain, True)

        delexicalizer.is_train = False
        delexicalizer.delex_domain(domain, False)
        delexicalizer.nlp.__del__()
        time.sleep(2)

    globaly_delex_adj(domains)


if __name__ == "__main__":
    domains = ['publications', 'restaurants', 'housing', 'recipes', 'socialnetwork', 'calendar', 'blocks']
    delex_domains(domains)
