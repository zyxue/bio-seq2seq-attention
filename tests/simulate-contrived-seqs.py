import string
import logging
import json

import numpy as np
from tqdm import tqdm

from seq2seq.objs import Language


logging.basicConfig(
    level=logging.DEBUG, format='%(asctime)s|%(levelname)s|%(message)s')

logger = logging.getLogger(__name__)


def dump_lang_config(lang0, lang1, output):
    cfg = {
        'lang0': lang0.to_dict(),
        'lang1': lang1.to_dict(),
    }
    with open(output, 'wt') as opf:
        json.dump(cfg, opf)


def sumprod(tokens0, seq_len, lang1):
    """
    At 1st half of tokens0, add right neighbour number to the output.
    At 2nd haf of tokens0, product with the corresponding token towards the end

    :params tokens0: a list of tokens
    :return tokens1: a list of tokens
    """
    tokens1 = []
    for i in range(seq_len):
        a = int(tokens0[i])
        if i < seq_len // 2:
            b = int(tokens0[i + 1])
            idx = a + b
        else:
            b = int(tokens0[seq_len - 1 - i])
            idx = a * b
        tokens1.append(lang1.index2token[idx])
    return tokens1


def simulate(lang0, lang1, num_seqs, seq_len, method, output_file):
    with open(output_file, 'wt') as opf:
        for i in tqdm(range(num_seqs)):
            tks0 = np.random.choice(lang0.vocab, seq_len)
            tks1 = eval(method)(tks0, seq_len, lang1)
            seq0 = ''.join(tks0)
            seq1 = ''.join(tks1)
            opf.write(f'{seq0} {seq1}\n')


def main():
    vocab0 = list(map(str, range(10)))
    vocab1 = string.ascii_letters + string.digits + string.punctuation
    # exclude reserved tokens
    vocab1 = [_ for _ in vocab1 if _ not in ['^', '$', '*']]

    lang0 = Language('lang0', vocab0)
    lang1 = Language('lang1', vocab1)

    logger.info(lang0)
    logger.info(lang1)

    dump_lang_config(lang0, lang1, './lang_config.json')

    # TODO: add argparse
    num_seqs = 10000
    seq_len = 10
    method = 'sumprod'
    out = f'./{method}_num_seqs_{num_seqs}_seq_len_{seq_len}.csv'
    logger.info(f'simulating {num_seqs} seq pairs using {method} ...')
    simulate(lang0, lang1, num_seqs, seq_len, method, out)


if __name__ == "__main__":
    main()
