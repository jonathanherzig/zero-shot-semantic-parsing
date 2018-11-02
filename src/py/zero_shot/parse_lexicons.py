import os
import pyparsing
import paths_utils

LEX_PATH = paths_utils.LEX_PATH


def parse_lexicon(lex_path, domain):
    """
    Preprocess the lexicon in its Overnight template to a more convenient one.
    :param lex_path: The path for the Overnight lexicons
    :param domain: Domain lexicon to parse
    :return:
    """
    parser = pyparsing.nestedExpr('(', ')', ignoreExpr=pyparsing.dblQuotedString)
    kb_constants = set()
    out_path = os.path.join(lex_path, domain + '.grammar.proc')

    for row in open(os.path.join(lex_path, domain + '.grammar')):
        row_strip = row.strip()
        try:
            parse = parser.parseString(row_strip)
            parse = parse[0]
            if parse[0] != 'rule':
                continue
            if parse[1] == '$TypeNP':
                nl = tuple(parse[2])
                lf = parse[3][1]
                name = ' '.join(nl)
                symbol = lf
                kb_constants.add((name, symbol, paths_utils.LF_ENTITY_TYPE))
                continue
            if parse[1] == '$EntityNP1' or parse[1] == '$EntityNP2':  # parse entities
                nl = tuple(parse[2])
                lf = parse[3][1]
                if not isinstance(lf, str):
                    if lf[0] == 'number':  # parse numbers
                        name = ' '.join(nl[1:])
                        symbol = lf[2]
                        kb_constants.add((name, symbol, paths_utils.LF_ENTITY_NUM))
                        continue
                    else:
                        continue
                else:
                    name = ' '.join(nl)
                    symbol = lf
                    kb_constants.add((name, symbol, paths_utils.LF_ENTITY))
            else:  # parse relations
                name = ' '.join(tuple(parse[2]))
                symbol = parse[3][1][1]
                if parse[1] == '$Rel0NP':  # the relation subject incase of neo-Davidsonian treatment of events
                    kb_constants.add((name, symbol, paths_utils.LF_REL_SUBJ))
                elif domain == 'basketball' and symbol.startswith('num_'):
                    kb_constants.add((name, symbol, paths_utils.LF_REL_NUM))
                elif domain == 'basketball' and symbol == 'season':
                    kb_constants.add((name, symbol, paths_utils.LF_REL_DATE))
                elif domain == 'restaurants' and (symbol in ['star_rating', 'price_rating', 'reviews']):
                    kb_constants.add((name, symbol, paths_utils.LF_REL_NUM))
                elif domain == 'restaurants' and (
                    symbol in ['reserve', 'credit', 'outdoor', 'takeout', 'delivery', 'waiter', 'kids', 'groups']):
                    kb_constants.add((name, symbol, paths_utils.LF_REL_UNARY))
                elif domain == 'housing' and symbol == 'posting_date':
                    kb_constants.add((name, symbol, paths_utils.LF_REL_DATE))
                elif domain == 'housing' and (symbol in ['rent', 'size']):
                    kb_constants.add((name, symbol, paths_utils.LF_REL_NUM))
                elif domain == 'housing' and (
                    symbol in ['allows_cats', 'allows_dogs', 'has_private_bath', 'has_private_room']):
                    kb_constants.add((name, symbol, paths_utils.LF_REL_UNARY))
                elif domain == 'recipes' and symbol == 'posting_date':
                    kb_constants.add((name, symbol, paths_utils.LF_REL_DATE))
                elif domain == 'recipes' and (symbol in ['preparation_time', 'cooking_time']):
                    kb_constants.add((name, symbol, paths_utils.LF_REL_NUM))
                elif domain == 'publications' and symbol == 'publication_date':
                    kb_constants.add((name, symbol, paths_utils.LF_REL_DATE))
                elif domain == 'publications' and symbol == 'won_award':
                    kb_constants.add((name, symbol, paths_utils.LF_REL_UNARY))
                elif domain == 'socialnetwork' and (
                    symbol in ['birthdate', 'education_start_date', 'education_end_date', 'employment_start_date',
                               'employment_end_date']):
                    kb_constants.add((name, symbol, paths_utils.LF_REL_DATE))
                elif domain == 'socialnetwork' and symbol == 'height':
                    kb_constants.add((name, symbol, paths_utils.LF_REL_NUM))
                elif domain == 'socialnetwork' and symbol == 'logged_in':
                    kb_constants.add((name, symbol, paths_utils.LF_REL_UNARY))
                elif domain == 'calendar' and symbol == 'date':
                    kb_constants.add((name, symbol, paths_utils.LF_REL_DATE))
                elif domain == 'calendar' and symbol == 'length':
                    kb_constants.add((name, symbol, paths_utils.LF_REL_NUM))
                elif domain == 'calendar' and (symbol in ['start_time', 'end_time']):
                    # kb_constants.add((name, symbol, 'relation_time'))
                    kb_constants.add((name, symbol, paths_utils.LF_REL_DATE))
                elif domain == 'calendar' and symbol == 'is_important':
                    kb_constants.add((name, symbol, paths_utils.LF_REL_UNARY))
                elif domain == 'blocks' and (symbol in ['length', 'width', 'height']):
                    kb_constants.add((name, symbol, paths_utils.LF_REL_NUM))
                elif domain == 'blocks' and symbol == 'is_special':
                    kb_constants.add((name, symbol, paths_utils.LF_REL_UNARY))
                else:
                    kb_constants.add((name, symbol, paths_utils.LF_REL))
        except pyparsing.ParseException:
            continue
    with open(out_path, 'wb') as f:
        for const in kb_constants:
            f.write('\t'.join(const)+'\n')


domains = ['publications', 'restaurants', 'housing', 'recipes', 'socialnetwork', 'calendar', 'blocks']
for domain in domains:
    parse_lexicon(lex_path=LEX_PATH, domain=domain)
