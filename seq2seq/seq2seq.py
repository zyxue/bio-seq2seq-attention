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
        'outdir'
    ]:
        logger.info(f'    {attr}:\t{getattr(args, attr)}')


def gen_langs(lang_config_json):
    with open(lang_config_json, 'rt') as inf:
        dd = json.load(inf)

    lang0 = Language(**dd['lang0'])
    lang1 = Language(**dd['lang1'])
    return lang0, lang1


def main():
    options = parse_args()

    device = U.get_device(options.device)
    logger.info(f'found device: {device}')

    log_args(args)

    logger.info(f'loading languages from {options.config}')
    lang0, lang1 = gen_langs(options.config)

    enc = encoder.EncoderRNN(
        lang0,
        options.embedding_dim,
        options.hidden_size,
        options.num_hidden_layers,
        options.bidirectional,
    )
    enc = enc.to(device)
    logger.info(f'encoder => \n{enc}')

    num_directions = 2 if options.bidirectional else 1

    dec = MLP

    if options.architecture == "encoder-decode":
        dec = decoder.AttnDecoderRNN(lang1,
                                     options.embedding_dim,
                                     # adjust decoder architecture accordingly based on num_directions
                                     options.hidden_size * num_directions,
                                     options.num_hidden_layers,
                                     dropout_p=0.1)
    elif options.architecture == "RNN+MLP":
        dec = decoder.MLPDecoder(lang1,
                                 options.hidden_size * num_directions,
                                 options.num_hidden_layers)

    dec.to(device)
    logger.info(f'decoder => \n{dec}')

    logger.info('start training ...')
    hist = train.train(encoder=enc,
                       decoder=dec,
                       data_file=options.data_file,
                       n_iters=options.num_iters,
                       batch_size=options.batch_size,
                       device=device,
                       tf_ratio=options.teacher_forcing_ratio,
                       lr=options.learning_rate,
                       print_loss_interval=options.print_loss_interval,
                       plot_attn_interval=options.plot_attn_interval,
                       architecture=options.architecture)

    os.makedirs(options.outdir, exist_ok=True)
    hist_out = os.path.join(options.outdir, 'hist.csv')
    logger.info(f'writing {hist_out} ...')
    with open(hist_out, 'wt') as opf:
        opf.write('iter,loss\n')
        for k, i in enumerate(hist):
            opf.write(f'{options.print_loss_interval * (k+1)},{i}\n')


if __name__ == "__main__":
    main()
