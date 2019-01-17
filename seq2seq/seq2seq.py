import os
import json
import logging

from seq2seq import encoder
from seq2seq import decoder
from seq2seq import train
from seq2seq.args import parse_args
from seq2seq.objs import Language
from seq2seq import utils as U


logging.basicConfig(
    level=logging.DEBUG, format='%(asctime)s|%(levelname)s|%(message)s')

logger = logging.getLogger(__name__)


def log_args(args):
    for attr in [
        'data_file',
        'embedding_dim',
        'hidden_size',
        'batch_size',
        'num_hidden_layers',
        'bidirectional',
    ]:
        logger.info(f'    {attr}:\t{getattr(args, attr)}')


def gen_langs(lang_config_json):
    with open(lang_config_json, 'rt') as inf:
        dd = json.load(inf)

    lang0 = Language(**dd['lang0'])
    lang1 = Language(**dd['lang1'])
    return lang0, lang1


def main():
    args = parse_args()

    device = U.get_device(args.device)
    logger.info(f'found device: {device}')

    log_args(args)

    logger.info(f'loading languages from {args.config}')
    lang0, lang1 = gen_langs(args.config)

    enc = encoder.EncoderRNN(
        lang0,
        args.embedding_dim,
        args.hidden_size,
        args.num_hidden_layers,
        args.bidirectional,
    )
    enc = enc.to(device)
    logger.info(f'encoder => \n{enc}')

    num_directions = 2 if args.bidirectional else 1
    dec = decoder.AttnDecoderRNN(
        lang1,
        args.embedding_dim,
        # adjust decoder architecture accordingly based on num_directions
        args.hidden_size * num_directions,
        args.num_hidden_layers,
        dropout_p=0.1
    )
    dec.to(device)
    logger.info(f'decoder => \n{dec}')

    logger.info('start training ...')
    hist = train.train(encoder=enc,
                       decoder=dec,
                       data_file=args.data_file,
                       n_iters=args.num_iters,
                       batch_size=args.batch_size,
                       device=device,
                       tf_ratio=args.teacher_forcing_ratio,
                       lr=args.learning_rate,
                       print_loss_interval=args.print_loss_interval,
                       plot_attn_interval=args.plot_attn_interval)

    hist_out = os.path.join(args.outdir, 'hist.csv')
    logger.info('writing {hist_out} ...')
    with open(hist_out, 'wt') as opf:
        opf.write('iter,batch_loss\n')
        for k, i in enumerate(hist):
            opf.write(f'{args.print_loss_interval * (k+1)},{i}\n')


if __name__ == "__main__":
    main()
