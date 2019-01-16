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


def tensor_from_sentence(lang, sentence):
    indexes = [lang.word2index[i] for i in sentence]
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)


def tensors_from_pair(lang0, lang1, pair):
    src_tensor = tensor_from_sentence(lang0, pair[0])
    tgt_tensor = tensor_from_sentence(lang1, pair[1])
    seq_len = pair[2]
    return (src_tensor, tgt_tensor, seq_len)


def get_device(s=None):
    if s is None:
        if torch.cuda.is_available():
            s = 'cuda:0'
        else:
            s = 'cpu'
    return torch.device(s)
