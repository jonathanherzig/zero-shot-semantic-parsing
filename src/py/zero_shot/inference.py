import operator
import os
import subprocess
import tempfile
from itertools import combinations
import heapq

import numpy as np

import attention_utils
import paths_utils
from delex_data import Delexicalizer


class Predictor(object):

    def __init__(self, domain):

        self.domain = domain
        self.delexicalizer = Delexicalizer(domain)
        self.cache = {}

    def score_derivation(self, x_toks, x_delex_toks, y_delex_toks, attention):
        """
        Find best assignment for an abstract logical form
        :param x_toks: lexical NL
        :param x_delex_toks: abstract NL
        :param y_delex_toks: predicted abstract logical form
        :param attention:
        :return:
        """

        res = self.cache.get(' '.join(x_toks) + '||||' + ' '.join(y_delex_toks))
        if res is not None:
            print 'found in cache'
            return res

        delexes = []

        y_delex_ = y_delex_toks
        pre_detected_toks = dict()
        pre_detected_toks['_number_'] = []
        pre_detected_toks['_date_'] = []
        pre_detected_toks['_time_'] = []
        pre_detected_toks['_entity_'] = []

        for i, y_tok in enumerate(y_delex_):
            if y_tok in paths_utils.LF_TOKS:
                delexes.append((y_tok, i))

        if len(delexes) == 0:
            return ' '.join(y_delex_toks), False, None

        global_cands_beam = []
        for delex in delexes:  # aggragete local candidates for all special tokens
            y_delex = delex[0]
            y_ind = delex[1]
            attention_mask = attention_utils.get_mask(y_delex, x_delex_toks, self.delexicalizer)
            # softmax
            exp_attention = np.exp(attention[:, y_ind])
            alignment_scores = attention_mask * exp_attention
            alignment_scores /= np.sum(alignment_scores)
            if y_delex in pre_detected_toks:
                alignment_scores_masked = alignment_scores * attention_mask
                sucess = self.handle_pre_detected_tok_soft(y_delex, pre_detected_toks, alignment_scores_masked, x_toks, y_ind, global_cands_beam)
                if not sucess:
                    return ' '.join(y_delex_toks), False, None
                continue

            cands = self.delexicalizer.lexicon.get(y_delex)
            if cands is None:
                return ' '.join(y_delex_toks), False, None

            score_embed_pairs = []
            for i in range(len(alignment_scores)):
                alignment_score = alignment_scores[i]
                if alignment_score > 0.0:
                    x_token_emb = self.delexicalizer.get_embedding(x_toks[i])
                    if x_token_emb is not None:
                        score_embed_pairs.append((alignment_score, x_token_emb))
            if len(score_embed_pairs) == 0:
                return ' '.join(y_delex_toks), False, None
            cands_sorted = attention_utils.get_similar_from_schema_multiple(cands, self.delexicalizer, score_embed_pairs)
            if len(cands_sorted) == 0:
                return ' '.join(y_delex_toks), False, None
            cands_sorted_beam = [([item[0]], item[1]) for item in cands_sorted]
            global_cands_beam.append(cands_sorted_beam)

        restrictions = self.get_restrictions(delexes, pre_detected_toks, x_toks, x_delex_toks)

        # exact_search
        top_size = 500
        cands = []
        horizon = []
        past_pushes = []
        current_assign = [0] * len(global_cands_beam)
        assign = self.get_score_of_assign(current_assign, global_cands_beam)
        cands.append(assign)
        while len(cands) < top_size:
            for index, value in enumerate(current_assign):
                if len(global_cands_beam[index]) > value+1:
                    next_assign = list(current_assign)
                    next_assign[index] = value + 1
                    if next_assign not in past_pushes:
                        assign_cands, assign_score = self.get_score_of_assign(next_assign, global_cands_beam)
                        heapq.heappush(horizon, (assign_score, (next_assign, assign_cands)))
                        past_pushes.append(next_assign)
            if len(horizon) == 0:  # no more assingmets to try
                break
            next = heapq.heappop(horizon)
            current_assign = next[1][0]
            cand = (next[1][1], next[0])
            cands.append(cand)
        if restrictions is not None:
            for j, cand in reversed(list(enumerate(cands))):
                if not self.enforce_restrictions(restrictions, cand):
                    del cands[j]
        y_sorted_ = [self.get_lexicalized_lf(y_delex_, delexes, assign[0]) for assign in cands]

        all_lfs_ = [self.format_lf(lf) for lf in y_sorted_]

        tf_lines = all_lfs_
        temp_file = tempfile.NamedTemporaryFile(suffix='.examples')
        for line in tf_lines:
            print >> temp_file, line
            # print line
        temp_file.flush()
        FNULL = open(os.devnull, 'w')
        msg = subprocess.check_output(['evaluator/overnight', self.domain, temp_file.name], stderr=FNULL)
        temp_file.close()
        denotations = [line.split('\t')[1] for line in msg.split('\n')
                       if line.startswith('targetValue\t')]

        y_exec = None
        good = 0
        succ_ind = None
        for i, den in enumerate(denotations):
            if not self.is_error(den):
                if y_exec is None:
                    y_exec = y_sorted_[i]
                    succ_ind = i+1
                good += 1
        if len(denotations) == 0:
            self.cache[' '.join(x_toks) + '||||' + ' '.join(y_delex_toks)] = (' '.join(y_delex_toks), False, succ_ind)
            return ' '.join(y_delex_toks), False, succ_ind
        if y_exec is None:
            self.cache[' '.join(x_toks) + '||||' + ' '.join(y_delex_toks)] = (y_sorted_[0], False, succ_ind)
            return y_sorted_[0], False, succ_ind

        self.cache[' '.join(x_toks) + '||||' + ' '.join(y_delex_toks)] = (y_exec, True, succ_ind)
        return y_exec, True, succ_ind

    def check_cand(self, cand, restrictions):
        if restrictions is not None:
            if not self.enforce_restrictions(restrictions, cand):
                    return False
        return True

    def get_restrictions(self, delexes, pre_detected_toks, x_toks, x_delex_toks):

        predetected_x_count = dict() # count how many times each object appears in NL
        for predetected_tok in pre_detected_toks:
            instances = set([attention_utils.argmax_y(delex_tok, tok,self.delexicalizer) for tok, delex_tok in zip(x_toks, x_delex_toks) if delex_tok == predetected_tok])
            predetected_x_count[predetected_tok] = len(instances)

        indices_lists = []
        restricted_tokens = pre_detected_toks.keys() + ['_relation_unary_']
        for pre_detect_tok in restricted_tokens:

            indices = [i for i, x in enumerate(delexes) if x[0] == pre_detect_tok]
            if len(indices) > 1:
                indices_lists += list(combinations(indices, 2))
        if len(indices_lists) > 0:
            return indices_lists
        else:
            return None

    def enforce_restrictions(self, restrictions, candidate):
        preds = candidate[0]
        for restriction in restrictions:
            left_ind = restriction[0]
            right_ind = restriction[1]
            if len(preds) > left_ind and len(preds) > right_ind and preds[restriction[0]] == preds[restriction[1]]:
                    return False
        return True

    def handle_pre_detected_tok_soft(self, y_delex, pre_detected_toks, alignment_scores, x_toks, y_ind, global_cands_beam):
        scores = dict()
        for i in range(len(alignment_scores)):
            alignment_score = alignment_scores[i]
            if alignment_score > 0.0:
                y_lex = attention_utils.argmax_y(y_delex, x_toks[i], self.delexicalizer)
                if y_lex is not None:
                    if y_delex in ['_time_', '_date_']:
                        y_lex = y_lex.replace('_', ' ')
                    scores[y_lex] = alignment_score
        if len(scores) == 0:
            return False
        scores_sorted = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        cands_sorted_beam = [([item[0]], item[1]) for item in scores_sorted]
        global_cands_beam.append(cands_sorted_beam)
        return True

    def preprocess_time_2(self, x_delex_toks, y_pred_delex_toks, pos, attention):
        x_delex_time_toks = list(x_delex_toks)
        pos_toks = pos.split()
        y_delex_time_toks = []
        relation_date_inds = []
        for i, pos_tok in enumerate(pos_toks):
            if pos_tok == paths_utils.NL_TIME:
                x_delex_time_toks[i] = paths_utils.NL_TIME
        for i, y_tok_delex in enumerate(y_pred_delex_toks):
            if y_tok_delex == '_date_':
                attention_mask = attention_utils.get_mask(y_tok_delex, x_delex_toks, self.delexicalizer)
                x_tok_lex, top_att_ind = attention_utils.argmax_x(x_delex_toks, attention[:, i], attention_mask)
                if x_tok_lex is None:
                    return None, None
                if pos_toks[top_att_ind] == paths_utils.NL_TIME:
                    y_delex_time_toks[-1] = 'time'
                    y_delex_time_toks.append('_time_')
                else:
                    y_delex_time_toks.append(y_tok_delex)
            else:
                y_delex_time_toks.append(y_tok_delex)
        return x_delex_time_toks, y_delex_time_toks

    def get_score_of_assign(self, assign, global_cands):
        score = 0
        cands = []
        for i, value in enumerate(assign):
            score += -1 * global_cands[i][value][1]  # negative for max heap implementation
            cands += global_cands[i][value][0]

        return (cands, score)

    def get_lexicalized_lf(self,y_delex_toks, delexes, assign):
        y_lex_toks = list(y_delex_toks)
        for i, delex in enumerate(delexes):
            ind = delex[1]
            y_lex_toks[ind] = assign[i]
        y_lex = ' '.join(y_lex_toks)
        return y_lex

    def format_lf(self, lf):
        # lf = self.postprocess_lf(lf)
        replacements = [
            ('! ', '!'),
            ('SW', 'edu.stanford.nlp.sempre.overnight.SimpleWorld'),
        ]
        for a, b in replacements:
            lf = lf.replace(a, b)
        # Balance parentheses
        num_left_paren = sum(1 for c in lf if c == '(')
        num_right_paren = sum(1 for c in lf if c == ')')
        diff = num_left_paren - num_right_paren
        if diff > 0:
            while len(lf) > 0 and lf[-1] == '(' and diff > 0:
                lf = lf[:-1]
                diff -= 1
            if len(lf) == 0: return ''
            lf = lf + ' )' * diff
        elif diff < 0: # extra left parentheses
            while len(lf) > 0 and lf[-1] == ')' and diff > 0:
                lf = lf[:-1]
                diff -= 1
        return lf

    def is_error(self, d):
        return 'BADJAVA' in d or 'ERROR' in d or d == 'null'
