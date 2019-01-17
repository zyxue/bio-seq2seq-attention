import time
import math
from datetime import timedelta
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


def time_since(since, percent):
    now = time.time()
    passed = int(now - since)

    # estimate total time
    total = passed / percent

    remained = int(total - passed)
    return f'time passed: {timedelta(seconds=passed)}, remaining: {timedelta(seconds=remained)}'


def train(encoder, decoder, data_file, n_iters, batch_size, device,
          lr, tf_ratio, print_loss_interval, plot_attn_interval):
    log_plot(plot_attn_interval)

    encoder_optim, decoder_optim = init_optimizers(encoder, decoder, lr)

    # TODO: remove assert, maybe add an pad_token to Language class
    assert decoder.language.unk_token_index == 2
    loss_func = nn.NLLLoss(
        # ignore padding tokens
        ignore_index=decoder.language.unk_token_index,
        # set to none because average will be taken over variable lengths
        reduction='none'
    )

    data_iter = prep_training_data(
        encoder.language, decoder.language, data_file, batch_size, device)

    loss_hist = []              # loss history
    loss_i = 0                  # loss total within print_loss_interval
    start = time.time()
    for idx in range(1, n_iters + 1):
        batch = next(data_iter)
        # loss_b: batch loss
        loss_b = train_on_one_batch(
            encoder, decoder, encoder_optim, decoder_optim,
            batch, loss_func, tf_ratio=0.5)
        loss_i += loss_b

        if idx % print_loss_interval == 0:
            loss_i /= print_loss_interval
            loss_hist.append(loss_i)

            percent = idx / n_iters
            logger.info(f'{time_since(start, percent)} {idx:d}/{n_iters}({percent:.3%}) iters: {loss_i:.4f}')
            loss_i = 0

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

    return loss_hist
