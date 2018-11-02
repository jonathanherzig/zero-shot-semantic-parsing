import os
import random
import paths_utils

DATA_IN_PATH = paths_utils.DATA_TEST_MODE_PATH
DATA_OUT_PATH = paths_utils.DATA_DEV_MODE_PATH


def write_domain(domain, train, dev):
    """
    Write train and validation for a domain. All other domains contribute their training data.
    :param domain: target domain
    :param train: dict with training data for all domains
    :param dev: dict with dev data for all domains
    :return:
    """
    with open(os.path.join(DATA_OUT_PATH, domain + '_test.tsv'), 'w') as f:
        for example in dev:
            f.write(example+'\n')

    with open(os.path.join(DATA_OUT_PATH, domain + '_train.tsv'), 'w') as f:
        for example in train:
            f.write(example + '\n')


def process_file(file, dev_frac):
    if not file.startswith('.'):
        domain = file.split('_')[0]

        is_test = file.split('_')[1] == 'test.tsv'
        if is_test:
            return

        data_raw = []
        for row in open(os.path.join(DATA_IN_PATH, file)):  # read training data
            example = row.rstrip()
            data_raw.append(example)
        print 'loaded {} train examples in domain {}'.format(len(data_raw), domain)

        num_dev = int(round(len(data_raw) * dev_frac))
        random.seed(0)
        random.shuffle(data_raw)
        dev = data_raw[:num_dev]
        train = data_raw[num_dev:]

        write_domain(domain, train, dev)


def create_dev_split(dev_frac=0.2):
    files = [f for f in os.listdir(DATA_IN_PATH) if os.path.isfile(os.path.join(DATA_IN_PATH, f))]
    for file in files:
        process_file(file, dev_frac)
