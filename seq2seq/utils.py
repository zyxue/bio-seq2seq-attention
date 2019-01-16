import time
import math

import torch
import numpy as np


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


def convert_seq2tensor(lang, seq):
    '''
    :param lang: a Language instance
    :param seq: the sequence string
    '''
    indexes = [lang.token2index[i] for i in seq]
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)


def get_device(s=None):
    if s is None:
        if torch.cuda.is_available():
            s = 'cuda:0'
        else:
            s = 'cpu'
    return torch.device(s)
