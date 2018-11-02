"""
A dataset class for the aligner
"""
import numpy as np


class Dataset():
    def __init__(self, x, y, similarities, batch_size):
        self.x = x
        self.y = y
        self.similarities = similarities
        self.curr_ind = 0
        self.batch_size = batch_size
        self.data_size = x.shape[0]

    def has_next_batch(self):
        return self.curr_ind != self.data_size

    def reset(self):
        self.curr_ind = 0

    def next_batch(self):
        end_ind = self.curr_ind + self.batch_size

        if end_ind > self.data_size:
            end_ind = self.data_size

        x_batch = self.x[self.curr_ind:end_ind, :]
        y_batch = self.y[self.curr_ind:end_ind, :]

        x_lengths = (x_batch != 0).cumsum(1).argmax(1) + 1
        y_lengths = (y_batch != 0).cumsum(1).argmax(1) + 1

        x_max_len = np.max(x_lengths)
        y_max_len = np.max(y_lengths)

        sims_batch = self.similarities[self.curr_ind:end_ind, :x_max_len, :y_max_len]

        x_batch = x_batch[:, :x_max_len]
        y_batch = y_batch[:, :y_max_len]

        self.curr_ind = end_ind

        return x_batch, y_batch, x_lengths, y_lengths, sims_batch
