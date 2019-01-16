import os
import pickle
import logging

from seq2seq import encoder
from seq2seq import decoder
from seq2seq import train
from seq2seq.args import parse_args
from seq2seq import utils as U


logging.basicConfig(
    level=logging.DEBUG, format='%(asctime)s|%(levelname)s|%(message)s')


def log_args(args):
    for attr in [
        'input_file',
        'embedding_dim',
        'hidden_size',
        'batch_size',
        'num_layers',
        'bidirectional',
    ]:
        logging.info(f'    {attr}:\t{getattr(args, attr)}')


def main():
    args = parse_args()

    device = U.get_device('cpu')
    logging.info(f'found device: {device}')

    log_args(args)

    infile = os.path.abspath(args.input_file)
    logging.info(f'reading {infile} ...')
    with open(infile, 'rb') as inf:
        # convention: lang0 is always the source language while lang1 is always
        # the target language
        lang0, lang1, seq_pairs = pickle.load(inf)

    logging.info(f'loaded {len(seq_pairs)} seqs')

    enc = encoder.EncoderRNN(
        lang0,
        args.embedding_dim,
        args.hidden_size,
        args.num_layers,
        args.bidirectional,
    )
    enc = enc.to(device)
    logging.info(f'encoder => \n{enc}')

    num_directions = 2 if args.bidirectional else 1
    dec = decoder.AttnDecoderRNN(
        lang1,
        args.embedding_dim,
        # adjust decoder architecture accordingly based on num_directions
        args.hidden_size * num_directions,
        args.num_layers,
        dropout_p=0.1
    )
    dec.to(device)
    logging.info(f'decoder => \n{dec}')

    logging.info('start training ...')
    hist = train.train_iters(
        lang0,
        lang1,
        enc,
        dec,
        lang1.token2index[lang1.beg_token],
        args.num_iters,
        args.batch_size,
        args.print_every,
        args.plot_every,
        args.learning_rate
    )


if __name__ == "__main__":
    main()
