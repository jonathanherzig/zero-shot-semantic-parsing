import os
import paths_utils


def process_domain(domain, data_in_path):
    """
    Get train and validation data for a domain
    """
    data_raw = []
    for row in open(os.path.join(data_in_path, domain + '_train.tsv')):   # read training data
        example = row.rstrip()
        data_raw.append(example)
    print 'loaded {} train examples in domain {}'.format(len(data_raw), domain)

    train = data_raw
    dev = []
    for row in open(os.path.join(data_in_path, domain + '_test.tsv')):
        example = row.rstrip()
        dev.append(example)
    print 'loaded {} dev examples in domain {}'.format(len(dev), domain)
    return train, dev


def write_domain(domain, domains, train, dev, data_out_path):
    """
    Write train and validation for a domain. All other domains contribute their training data.
    :param domain: target domain
    :param domains: all domains
    :param train: dict with training data for all domains
    :param dev: dict with dev data for all domains
    :return:
    """
    with open(os.path.join(data_out_path, domain + '_test.tsv'), 'w') as f:
        for example in dev[domain]:
            f.write(example+'\n')

    with open(os.path.join(data_out_path, domain + '_train.tsv'), 'w') as f:
        for d in domains:
            if d != domain:
                for example in train[d]:
                    f.write(example + '\n')


def process_cross_domain(domains, is_split_test):

    if is_split_test:
        data_in_path = paths_utils.DATA_TEST_MODE_PATH
        data_out_path = paths_utils.DATA_TEST_MODE_CROSS_PATH
    else:
        data_in_path = paths_utils.DATA_DEV_MODE_PATH
        data_out_path = paths_utils.DATA_DEV_MODE_CROSS_PATH

    train = dict()
    dev = dict()
    domains = sorted(domains)
    for domain in domains:
        train_, dev_ = process_domain(domain, data_in_path)
        train[domain] = train_
        dev[domain] = dev_
    for domain in domains:
        write_domain(domain, domains, train, dev, data_out_path)


if __name__ == "__main__":
    domains = ['publications', 'restaurants', 'housing', 'recipes', 'socialnetwork', 'calendar', 'blocks']
    process_cross_domain(domains, is_split_test=True)


