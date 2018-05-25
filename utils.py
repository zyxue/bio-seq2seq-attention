import time
import math

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
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


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def calc_accuracy(seq1, seq2):
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))
    return (np.array(list(seq1))[:min_len] ==
            np.array(list(seq2))[:min_len]).sum() / max_len
