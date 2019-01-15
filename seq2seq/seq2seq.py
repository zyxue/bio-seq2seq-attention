import os
import pickle

from bioseq2seq import encoder
from bioseq2seq import decoder
from bioseq2seq import train
from bioseq2seq import utils as U


DEVICE = U.get_device()
print('device: {0}'.format(DEVICE))


if __name__ == "__main__":
    args = U.parse_args()

    data_file = args.input

    for i in [
        'embedding_size',
        'hidden_size',
        'batch_size',
        'num_layers',
        'bidirectional',
    ]:
        print('{0}:\t{1}'.format(i, getattr(args, i)))

    print('reading data from {0}'.format(os.path.abspath(data_file)))
    with open(data_file, 'rb') as inf:
        src_lang, tgt_lang, pairs = pickle.load(inf)

    # print('filter out seqs longer than {0}'.format(MAX_LENGTH))
    # pairs = [_ for _ in pairs if _[-1] <= MAX_LENGTH]

    print('data loaded...')
    print('training on {0} seqs'.format(len(pairs)))

    sos_symbol = '^'            # symbol for start of a seq
    tgt_sos_index = tgt_lang.word2index['^']

    enc = encoder.EncoderRNN(
        src_lang.n_words,
        args.embedding_size,
        args.hidden_size,
        args.num_layers,
        args.bidirectional,
    )
    enc = enc.to(DEVICE)

    num_directions = 2 if args.bidirectional else 1
    dec = decoder.AttnDecoderRNN(
        args.embedding_size,
        # adjust decoder architecture accordingly based on num_directions
        args.hidden_size * num_directions,
        args.num_layers,
        tgt_lang.n_words,
        dropout_p=0.1
    )
    dec.to(DEVICE)

    hist = train.train_iters(
        src_lang,
        tgt_lang,
        enc,
        dec,
        tgt_sos_index,
        args.num_iters,
        args.batch_size,
        args.print_every,
        args.plot_every,
        args.learning_rate
    )
