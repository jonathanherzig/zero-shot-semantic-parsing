"""
Utils for using attention weights between abstract logical form tokens and abstract natural language words.
"""
import operator

import numpy as np

from aligner import Loaded_model
import paths_utils


def get_mask(y_tok_delex, x_toks_delex, delexicalizer):
    cands = delexicalizer.cands[y_tok_delex]
    mask = np.ones(len(x_toks_delex))
    for i, x_tok in enumerate(x_toks_delex):
        if x_tok not in cands:
            mask[i] = 0.0
    return mask


def argmax_x(x_toks, attention, mask):
    attention_masked = mask * attention
    if not attention_masked.any():  # if all zeros return None
        return None, None
    top_attention = np.argmax(attention_masked)
    return x_toks[top_attention], top_attention


def argmax_y(y_tok_delex, x_tok, delexicalizer):
    if y_tok_delex == '_number_':  # deal with a number candidate
        return deanonymize_number(x_tok, delexicalizer)
    elif y_tok_delex == '_date_':
        return deanonymize_date(x_tok)
    elif y_tok_delex == '_entity_': # return most similar entity, if any, according to edit distance
        return deanonymize_entity(x_tok, delexicalizer.lexicon[paths_utils.LF_ENTITY], delexicalizer)
    elif y_tok_delex == '_time_':
        return deanonymize_time(x_tok)

    x_embed = delexicalizer.get_embedding(x_tok)
    if x_embed is None:
        return None
    y_cands = delexicalizer.schema_map[y_tok_delex]
    y_sorted = get_similar_from_schema(y_cands, delexicalizer, x_embed)
    return y_sorted[0][0]


def get_similar_from_schema(candidates, delexicalizer, x_token_emb):
  dists = {}
  for predicate in candidates:
    dists[candidates[predicate]] = calc_pred_sim(x_token_emb, predicate, delexicalizer)
  dists_sorted = sorted(dists.items(), key=operator.itemgetter(1), reverse=True)
  return dists_sorted


def get_similar_from_schema_multiple(candidates, delexicalizer, score_embed_pairs):
  dists = {}
  for pair in score_embed_pairs:
      score = pair[0]
      x_embed = pair[1]
      for predicate in candidates:
          if candidates[predicate] not in dists:
              dists[candidates[predicate]] = 0
          dists[candidates[predicate]] += score * calc_pred_sim(x_embed, predicate, delexicalizer)
  dists_sorted = sorted(dists.items(), key=operator.itemgetter(1), reverse=True)
  return dists_sorted


def calc_pred_sim(x_embed, predicate, delexicalizer):
    pred_embed = get_pred_embed(predicate, delexicalizer)
    dist = calc_sim(x_embed, pred_embed)
    return dist


def get_pred_embed(predicate, delexicalizer):
    if predicate is None:
        return None
    embeds = []
    for token in predicate:
        token_embed = delexicalizer.get_embedding(token)
        if token_embed is not None:
            embeds.append(token_embed)
    pred_embed = np.mean(np.array(embeds), axis=0)
    return pred_embed


def calc_sim(e_1, e_2):
    dot = np.sum(e_1 * e_2)
    s_1 = np.sqrt(np.sum(e_1 ** 2))
    s_2 = np.sqrt(np.sum(e_2 ** 2))
    s_d = 1 - dot/(s_1 * s_2)
    sim = 1 - s_d/2
    return sim


def deanonymize_entity(x_tok, entities, delexicalizer):
    x_toks = x_tok.split('_')
    for entity in sorted(entities, key=len, reverse=True): # sort entities by length and find matches
        span = delexicalizer.find_entity_spans(x_toks, entity)
        if span is not None:
            return entities[entity]
    return None


def deanonymize_date(x_tok):
    from dateutil import parser
    try:
        dt = parser.parse(x_tok.replace('_', ' '))
        year = dt.year
        month = dt.month
        day = dt.day
        if year > 2015:
            return '2015' + '_' + str(month) + '_' + str(day)
        else:
            return str(year) + '_' + '-1_-1'
    except:
        return None


def deanonymize_time(x_tok):
    from dateutil import parser
    try:
        dt = parser.parse(x_tok.replace('_', ' '))
        hour = dt.hour
        return str(hour) + '_' + '0'
    except:
        return None


def deanonymize_number(x_tok, delexicalizer):
    try:
        y_tok = int(x_tok)
        return str(y_tok)
    except ValueError:
        y_tok = delexicalizer.word2num.get(x_tok)
        if y_tok is None:
            return None
        else:
            return str(y_tok)


def get_trained_model(domain):
    return Loaded_model(domain)


