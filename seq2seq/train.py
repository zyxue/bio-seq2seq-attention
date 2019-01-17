import time
import random
import time
import logging

import torch
import torch.nn as nn
from torch import optim
import numpy as np

from seq2seq.data import prep_training_data
from seq2seq.evaluate import evaluate_randomly
import seq2seq.utils as U


logger = logging.getLogger(__name__)


def encode(encoder, data_batch, encoder_optim):
    """
    run data through encoder and return the encoded tensors

    data_batch is a tuple of (seq0s, seq1s, seq_lens).
    seq0s and seq1s should be in indices
    """
    encoder_optim.zero_grad()

    seq0s, _, _ = data_batch
    # seq0s.shape: L x B
    _, B = seq0s.shape

    hid = encoder.init_hidden(B)
    out, hid = encoder(seq0s, hid)

    return out, hid


def init_beg_token(lang, batch_size):
    """
    for batch_size of 2, the return value will be like
    tensor([[0, 0]])
    """
    idx = lang.token2index[lang.beg_token]
    # repeat method in torch not found
    # reshape following convention: L x B
    arr = np.repeat([[idx]], batch_size, axis=1)
    return torch.tensor(arr)


def train(encoder, decoder, data_batch, encoder_optim, decoder_optim,
          criterion, teacher_forcing_ratio=0.5):
    enc_out, enc_hid = encode(encoder, data_batch, encoder_optim)

    decoder_optim.zero_grad()

    seq0s, seq1s, seq_lens = data_batch  # all in indices
    seq_len, batch_size = seq0s.shape

    # inherit the hidden state from the last step output from the encoder
    dec_hid = enc_hid

    # init the first output token for the decoder
    dec_in = init_beg_token(decoder.language, batch_size)

    use_tforce = True if random.random() < teacher_forcing_ratio else False

    loss = 0
    if use_tforce:
        for di in range(seq_len):
            dec_out, dec_hid, dec_attn = dec(dec_in, dec_hid, enc_outs)
            loss += criterion(dec_out, tgt_tensor[di].view(-1))
            dec_in = tgt_tensor[di]
    else:
        for di in range(seq_len):
            dec_out, dec_hid, dec_attn = dec(dec_in, dec_hid, enc_outs)
            topv, topi = dec_out.topk(1)
            # Returns a new Tensor, detached from the current graph. The result
            # will never require gradient.
            # https://pytorch.org/docs/stable/autograd.html#torch.Tensor.detach
            dec_in = topi.detach()
            loss += criterion(dec_out, tgt_tensor[di].view(-1))

    loss.backward()

    enc_optim.step()
    dec_optim.step()

    # TODO: better mask output padded seqs when calculating loss
    return loss.item() / seq_len.item()


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
                lr_update_every=2000, print_interval=1000, plot_interval=100):
    logger.info('Training for {0} steps'.format(n_iters))
    logger.info('Collect loss for plotting every {print_interval} steps')

    log_plot(plot_interval)

    data_iter = prep_training_data(encoder.language, decoder.language,
                                   data_file, batch_size)
    opm_enc, opm_dec = init_optimizers(encoder, decoder, lr)
    criterion = nn.NLLLoss()

    for k, i in enumerate(data_iter):
        print(k, i)
        if k == 100:
            break

    print_loss_total = 0
    print_losses = []

    # for idx in range(1, n_iters + 1):
    #     src_tensor, tgt_tensor, seq_lens = tr_pair_tensors[idx - 1]

    #     beg_tk_idx = lang1.token2index[lang1.beg_token],
    #     loss = train(
    #         lang0, lang1, enc, dec, src_tensor, tgt_tensor, seq_lens,
    #         lang1_beg_token_index, enc_optim, dec_optim, criterion,
    #         batch_size=batch_size
    #     )

    #     print_loss_total += loss

    #     if idx % print_every == 0:
    #         print_loss_avg = print_loss_total / print_every
    #         print_losses.append(print_loss_avg)
    #         print_loss_total = 0
    #         print('%s (%d %d%%) %.4f' % (
    #             time_since(start, idx / n_iters),
    #             idx,
    #             idx / n_iters * 100,
    #             print_loss_avg))

    #     if idx % plot_every == 0:
    #         evaluate_randomly(
    #             lang0, lang1, enc, dec, lang1_beg_token_index, 1, idx)

    #     if idx % lr_update_every == 0:
    #         enc_scheduler.step(loss)
    #         dec_scheduler.step(loss)
    #         print('iter {0}, enc lr: {1}, dec lr: {2}'.format(
    #             idx,
    #             enc_scheduler.optimizer.param_groups[0]['lr'],
    #             enc_scheduler.optimizer.param_groups[0]['lr']
    #         ))

    return print_losses
