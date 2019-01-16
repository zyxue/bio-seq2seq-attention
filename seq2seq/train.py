import time
import random
import time
import logging

import torch
import torch.nn as nn
from torch import optim

from seq2seq.evaluate import evaluate_randomly
import seq2seq.utils as U


logger = logging.getLogger(__name__)


def train(encoder, decoder, data_batch, encoder_optim, decoder_optim,
          criterion, teacher_forcing_ratio=0.5):
    lang1 = decoder.language
    beg_tk_idx = lang1.token2index[lang1.beg_token]

    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    # encoding source sequence
    seq0s, seq1s, seq_lens = data_batch  # all in indices
    batch_size = len(seq0s[0])

    enc_hid = encoder.init_hidden(batch_size)
    enc_outs, enc_hid = encoder(seq0s, enc_hid)

    import pdb; pdb.set_trace()

    if encoder.bidirectional:
        # as the enc_outs has a 2x factor for hidden size, so reshape hidden to
        # match that
        enc_hid = torch.cat([
            enc_hid[:encoder.num_layers, :, :],
            enc_hid[encoder.num_layers:, :, :]
        ], dim=2)

    # take the hidden state from the last step in the encoder, continue in the
    # decoder
    dec_hid = enc_hid
    # init the first input for the decoder
    dec_in = torch.tensor([[lang1_beg_token_index] * batch_size], device=device).view(-1, 1)

    # decide to use teacher forcing or not
    use_tf = True if random.random() < teacher_forcing_ratio else False

    loss = 0
    seq_len = max(seq_lens)
    if use_tf:
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


def pad_seqs(seqs):
    max_len = max(len(i) for i in seqs)

    # 2 corresponds to the unk_token. TODO: replace 2 with something more sensible
    seqs = [i + [2] * (max_len - len(i)) for i in seqs]
    return seqs


def convert_to_tensor(seqs):
    # should be of shape (seq_len, batch, 1) based on pytorch convention:
    # https://pytorch.org/docs/stable/nn.html#torch.nn.GRU
    return torch.tensor(seqs).transpose(1, 0)


def prep_training_data(lang0, lang1, data_file, batch_size):
    """
    prepare training data in tensors, returns an infinite generator

    :param seq_pairs: a list of (seq0, seq1, length) tuples
    """
    # assuming lines in data_file are already shuffled
    seq0s, seq1s, seq_lens, counter = [], [], [], 0
    while True:
        with open(data_file, 'rt') as inf:
            for line in inf:
                seq0, seq1 = line.strip().split()
                assert len(seq0) == len(seq1)

                seq0s.append(lang0.seq2indices(seq0))
                seq1s.append(lang1.seq2indices(seq1))
                seq_lens.append(len(seq0))
                counter += 1

                if counter == batch_size:
                    seq0s = convert_to_tensor(pad_seqs(seq0s))
                    seq1s = convert_to_tensor(pad_seqs(seq1s))
                    yield pad_seqs([seq0s, seq1s, seq_lens])
                    # reset
                    seq0s, seq1s, seq_lens, counter = [], [], [], 0


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
