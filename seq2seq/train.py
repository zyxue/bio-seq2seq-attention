import time
from datetime import timedelta
import logging

import torch
import torch.nn as nn
from torch import optim


from seq2seq.data import SeqData
from seq2seq.train_on_one_batch import train_on_one_batch


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


def train(encoder, decoder, data_file, num_iters, batch_size, lr,
          print_loss_interval):
    device = encoder.device
    encoder_optim, decoder_optim = init_optimizers(encoder, decoder, lr)

    # TODO: remove assert, maybe add an pad_token to Language class
    assert decoder.language.unk_token_index == 2
    loss_func = nn.NLLLoss(
        # ignore padding tokens
        ignore_index=decoder.language.unk_token_index,
        # set to none because average will be taken over variable lengths
        reduction='none'
    )

    data_iter = SeqData(data_file, device)
    data_len = len(data_iter)

    loss_hist = []              # loss history
    loss_i = 0                  # loss total within print_loss_interval
    start = time.time()
    for idx in range(1, num_iters + 1):
        batch = data_iter[(idx - 1) % data_len]
        _seq = batch[0].unsqueeze(dim=1)
        _lab = batch[1].unsqueeze(dim=1)
        _len = torch.tensor([_seq.shape[0]], device=device)
        batch = [_seq, _lab, _len]

        # loss_b: batch loss
        loss_b = train_on_one_batch(encoder, decoder, encoder_optim,
                                    decoder_optim, batch, loss_func)
        loss_i += loss_b

        if idx % print_loss_interval == 0:
            loss_i /= print_loss_interval
            loss_hist.append(loss_i)

            percent = idx / num_iters
            logger.info(f'{time_since(start, percent)} {idx:d}/{num_iters}({percent:.1%}) iters: {loss_i:.4f}')
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
