import time
import math
import argparse

import numpy as np


class Lang(object):
    def __init__(self, name, vocab_file):
        self.name = name
        self.vocab_file = vocab_file

        self.word2index = {}
        with open(vocab_file) as inf:
            for k, i in enumerate(inf):
                i = i.strip()
                if not i:       # empty line
                    continue
                self.word2index[i] = k

        self.index2word = {j: i for (i, j) in self.word2index.items()}

        self.n_words = len(self.word2index)

    def __repr__(self):
        return 'Lang("{0}", "{1}")'.format(self.name, self.vocab_file)

    def __str__(self):
        return 'Lang: {0}, {1} words'.format(self.name, self.n_words)


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def calc_accuracy(seq1, seq2):
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))
    return (np.array(list(seq1))[:min_len] ==
            np.array(list(seq2))[:min_len]).sum() / max_len


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-e', '--embedding-size', type=int)
    parser.add_argument('-d', '--hidden-size', type=int)
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('-l', '--num-layers', type=int, default=1)
    parser.add_argument('-r', '--bidirectional', type=bool, default=False)

    parser.add_argument('-t', '--num-iters', type=int, default=5000)
    parser.add_argument('-p', '--print-every', type=int, default=100)

    args = parser.parse_args()
    return args
