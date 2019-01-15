import time
import math

import torch
import numpy as np


class Lang(object):
    def __init__(self, name, vocab=None):
        self.name = name
        self.vocab = vocab      # should be a list

        self.word2index = {}

        # with open(vocab_file) as inf:
        #     for k, i in enumerate(inf):
        #         i = i.strip()
        #         if not i:       # empty line
        #             continue
        #         self.word2index[i] = k

        for k, i in enumerate(vocab):
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


def tensor_from_sentence(lang, sentence, device):
    indexes = [lang.word2index[i] for i in sentence]
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(src_lang, tgt_lang, pair):
    src_tensor = tensor_from_sentence(src_lang, pair[0])
    tgt_tensor = tensor_from_sentence(tgt_lang, pair[1])
    seq_len = pair[2]
    return (src_tensor, tgt_tensor, seq_len)


def get_device(s=None):
    if s is None:
        if torch.cuda.is_available():
            s = 'cuda:0'
        else:
            s = 'cpu'
    return torch.device(s)
