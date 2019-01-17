import logging

import torch.nn as nn
from torch import optim


from seq2seq.data import prep_training_data
from seq2seq.train_on_one_batch import train_on_one_batch

from seq2seq.evaluate import evaluate_randomly
import seq2seq.utils as U


logger = logging.getLogger(__name__)



def log_plot(plot_interval):
    if plot_interval > 0:
        logger.info(f'plot attention map every {plot_interval} steps')
    else:
        logger.warning(f'NO attention map will be plotted during training')


def init_optimizers(encoder, decoder, lr):
    opt0 = optim.Adam(encoder.parameters(), lr=lr)
    opt1 = optim.Adam(decoder.parameters(), lr=lr)
    return opt0, opt1


def train_iters(encoder, decoder, data_file, n_iters, batch_size=1, lr=0.01,
                lr_update_every=2000, tf_ratio=0.5,
                print_interval=1000, plot_interval=100):
    logger.info('Training for {0} steps'.format(n_iters))
    logger.info('Collect loss for plotting every {print_interval} steps')

    log_plot(plot_interval)

    data_iter = prep_training_data(encoder.language, decoder.language,
                                   data_file, batch_size)
    encoder_optim, decoder_optim = init_optimizers(encoder, decoder, lr)
    loss_func = nn.NLLLoss()

    print_loss_total = 0
    print_losses = []

    for idx in range(1, n_iters + 1):
        batch = next(data_iter)
        loss = train_on_one_batch(
            encoder, decoder, encoder_optim, decoder_optim,
            batch, loss_func, tf_ratio=0.5
        )

        # print_loss_total += loss

        # if idx % print_every == 0:
        #     print_loss_avg = print_loss_total / print_every
        #     print_losses.append(print_loss_avg)
        #     print_loss_total = 0
        #     print('%s (%d %d%%) %.4f' % (
        #         time_since(start, idx / n_iters),
        #         idx,
        #         idx / n_iters * 100,
        #         print_loss_avg))

        # if idx % plot_every == 0:
        #     evaluate_randomly(
        #         lang0, lang1, enc, dec, lang1_beg_token_index, 1, idx)

        # if idx % lr_update_every == 0:
        #     enc_scheduler.step(loss)
        #     dec_scheduler.step(loss)
        #     print('iter {0}, enc lr: {1}, dec lr: {2}'.format(
        #         idx,
        #         enc_scheduler.optimizer.param_groups[0]['lr'],
        #         enc_scheduler.optimizer.param_groups[0]['lr']
        #     ))

    return print_losses
