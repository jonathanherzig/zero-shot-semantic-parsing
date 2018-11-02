"""
Fast align utils for predicting unsupervised alignments over the lexical source domains
"""
import os
import subprocess

import paths_utils

DATA_IN_PATH = paths_utils.ALIGNMENTS_IN_PATH
DATA_OUT_PATH = paths_utils.ALIGNMENTS_OUT_PATH
FAST_ALIGN_PATH = paths_utils.FAST_ALIGN_PATH
TRAIN_DATA_PATH = paths_utils.DATA_PATH
DELIMITER = ' ||| '


def create_align_input(input_file):
    """
    Create fast_align format
    :param input_file: The file to be formatted
    """
    file_name = input_file.split('/')[-1]
    target_domain = file_name.split('_')[0]

    all_data_raw = []
    for row in open(input_file):   # read training data
        example = row.rstrip().split('\t')
        x = example[4]
        y = example[5]
        all_data_raw.append(x + DELIMITER + y + '\n')
    print 'loaded {} train examples from file {}'.format(len(all_data_raw), input)
    input_align_file = file_name + '.align'
    with open(os.path.join(DATA_IN_PATH, input_align_file), 'w') as f:
        for example in all_data_raw:
            f.write(example)
    return input_align_file, target_domain


def train_alignment(input, domain):
    output_for = domain+'.alignment'
    input_path = os.path.join(DATA_IN_PATH, input)
    output_for_path = os.path.join(DATA_OUT_PATH, output_for )
    command = FAST_ALIGN_PATH + ' -i ' + input_path + ' -v  > ' + output_for_path
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print process.returncode

    output_cond_prob_back = domain + '.cond_prob'
    input_path = os.path.join(DATA_IN_PATH, input)
    command = FAST_ALIGN_PATH + ' -i ' + input_path + ' -v -r -p ' + os.path.join(DATA_OUT_PATH, output_cond_prob_back)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print process.returncode

    return output_for, output_cond_prob_back


def load_alignment(domains):
    outputs = []
    for domain in domains:
        output_path = os.path.join(DATA_OUT_PATH, domain+'.alignment')
        for row in open(output_path):  # read alignments
            alignments = [(int(pair.split('-')[0]), int(pair.split('-')[1])) for pair in row.rstrip().split(' ')]
            outputs.append(alignments)
    return outputs


def load_cond_prob(cond_prob_output):
    output_path = os.path.join(DATA_OUT_PATH, cond_prob_output)
    mapping = dict()
    for row in open(output_path):  # read alignments
        items = row.rstrip().split('\t')
        if cond_prob_output.endswith('for'):
            lf = items[1]
            nl = items[0]
        else:
            lf = items[0]
            nl = items[1]
        score = items[2]

        if lf not in mapping:
            mapping[lf] = dict()
        mapping[lf][nl] = float(score)
    return mapping


def load_cond_probs(cond_prob_outputs, domains):
    mappings = {}
    for (cond_prob_output, domain) in zip(cond_prob_outputs, domains):
        mappings[domain] = load_cond_prob(cond_prob_output)
    return mappings


def inspect_alignment_file(input, output):
    input_path = os.path.join(DATA_IN_PATH, input)
    output_path = os.path.join(DATA_OUT_PATH, output)

    inputs = []
    outputs = []

    for row in open(input_path):   # read training data
        x, y = row.rstrip().split(DELIMITER)
        inputs.append((x, y))

    for row in open(output_path):   # read alignments
        outputs.append(row.rstrip().split(' '))

    for i, example in enumerate(inputs):
        inspect_alignment(example[0], example[1], outputs[i])


def inspect_alignment(x, y, alignment):
    x_toks = x.split()
    y_toks = y.split()

    align_map = {}
    for a in alignment:
        x_ind = a.split('-')[0]
        y_ind = a.split('-')[1]
        align_map[int(y_ind)] = int(x_ind)

    print x
    print y
    for i, y_tok in enumerate(y_toks):
        align = align_map.get(i)
        x_ind
        if align is None:
            x_tok = 'NO_ALIGN'
            x_ind = '_'
        else:
            x_tok = x_toks[align]
            x_ind = align
        print str(i) + '\t' + y_tok + '\t' + str(x_ind) + '\t' + x_tok
    print ' '


if __name__ == "__main__":
    domains = ['publications', 'restaurants', 'housing', 'recipes', 'socialnetwork', 'calendar', 'blocks']
    for domain in domains:
        input_file = os.path.join(TRAIN_DATA_PATH, domain+'_train.tsv')
        input_align_file, target_domain = create_align_input(input_file)
        train_alignment(input_align_file, domain+'_train')

    for domain in domains:
        input_file = os.path.join(TRAIN_DATA_PATH, domain + '_test.tsv')
        input_align_file, target_domain = create_align_input(input_file)
        train_alignment(input_align_file, domain + '_test')
