import theano
import numpy as np

class DomainStatsVocab:

    def __init__(self, dataset, is_use):
        self.emb_mat, self.emb_size, self.domain_to_index = self.process_dataset(dataset)
        self.is_use = is_use

    def process_dataset(self, dataset):
        domain_to_stats = dict((x[6], x[8]) for x in dataset)
        domain_to_index = dict()
        emb_mat_ = []
        for i, domain in enumerate(domain_to_stats):
            domain_to_index[domain] = i
            emb_mat_.append([float(x) for x in domain_to_stats[domain].split(',')])
        emb_mat_ = np.array(emb_mat_)
        emb_size = emb_mat_.shape[1]
        emb_mat_ = emb_mat_.astype(theano.config.floatX)
        emb_mat = theano.shared(
            name='domain_emb_mat',
            value=emb_mat_)
        return emb_mat, emb_size, domain_to_index

    def get_theano_embedding(self, i):
        """Get theano embedding for given word index."""
        return self.emb_mat[i]
