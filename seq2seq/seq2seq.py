import os
import json
import logging

from seq2seq.encoders import EncoderRNN
from seq2seq.decoders import MLPDecoder
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

    log_args(options)

    logger.info(f'loading languages from {options.config}')
    lang0, lang1 = gen_langs(options.config)

    enc = EncoderRNN(lang0,
                     options.embedding_dim,
                     options.hidden_size,
                     options.num_hidden_layers,
                     device=device)
    logger.info(f'encoder => \n{enc}')

    num_directions = 2          # bidirectional

    dec = MLPDecoder(lang1,
                     options.hidden_size * num_directions,
                     options.num_hidden_layers,
                     device=device)
    logger.info(f'decoder => \n{dec}')

    logger.info('start training ...')
    hist = train.train(enc,
                       dec,
                       options.data_file,
                       options.num_iters,
                       options.batch_size,
                       options.learning_rate,
                       options.print_loss_interval,
                       device=device)

    os.makedirs(options.outdir, exist_ok=True)
    hist_out = os.path.join(options.outdir, 'hist.csv')
    logger.info(f'writing {hist_out} ...')
    with open(hist_out, 'wt') as opf:
        opf.write('iter,loss\n')
        for k, i in enumerate(hist):
            opf.write(f'{options.print_loss_interval * (k+1)},{i}\n')


if __name__ == "__main__":
    main()
