import time
import random

import torch
import torch.nn as nn
from torch import optim

from seq2seq.evaluate import evaluate_randomly
from seq2seq.utils import tensors_from_pair, time_since


def train(src_lang, tgt_lang, enc, dec, src_tensor, tgt_tensor, seq_lens,
          tgt_sos_index, enc_optim, dec_optim,
          criterion, device, teacher_forcing_ratio=0.5, batch_size=1):
    enc_optim.zero_grad()
    dec_optim.zero_grad()

    # encoding source sequence
    enc_hid = enc.init_hidden(batch_size)
    enc_outs, enc_hid = enc(src_tensor, enc_hid)

    if enc.bidirectional:
        # as the enc_outs has a 2x factor for hidden size, so reshape hidden to
        # match that
        enc_hid = torch.cat([
            enc_hid[:enc.num_layers, :, :],
            enc_hid[enc.num_layers:, :, :]
        ], dim=2)

    # take the hidden state from the last step in the encoder, continue in the
    # decoder
    dec_hid = enc_hid
    # init the first input for the decoder
    dec_in = torch.tensor([[tgt_sos_index] * batch_size], device=device).view(-1, 1)

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


def train_iters(
        src_lang, tgt_lang, enc, dec, tgt_sos_index, n_iters,
        batch_size=1,
        print_every=1000,
        plot_every=100,
        learning_rate=0.01,
        lr_update_every=2000,
):
    print('Training for {0} steps'.format(n_iters))
    print('Collect loss for plotting every {0} steps'.format(print_every))

    if plot_every > 0:
        print('Plot attention map every {0} steps'.format(plot_every))
    else:
        print('No attention map will be plotted during training')

    start = time.time()
    print_loss_total = 0
    print_losses = []

    # enc_optim = optim.SGD(enc.parameters(), lr=learning_rate)
    # dec_optim = optim.SGD(dec.parameters(), lr=learning_rate)
    enc_optim = optim.Adam(enc.parameters(), lr=learning_rate)
    dec_optim = optim.Adam(dec.parameters(), lr=learning_rate)
    enc_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        enc_optim, 'min', min_lr=0.0001)
    dec_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        dec_optim, 'min', min_lr=0.0001)

    tr_pair_tensors = []
    for i in range(n_iters):
        src_tsrs, tgt_tsrs, seq_lens = [], [], []
        for j in range(batch_size):
            _s, _t, _l = tensors_from_pair(
                src_lang, tgt_lang, random.choice(pairs))
            src_tsrs.append(_s)
            tgt_tsrs.append(_t)
            seq_lens.append(_l)
            # TODO: padding should happen here
        tr_pair_tensors.append([
            # make sure it's (seq_len, batch, input_size)
            torch.stack(src_tsrs, dim=1),
            torch.stack(tgt_tsrs, dim=1),
            torch.tensor(seq_lens),
        ])

    criterion = nn.NLLLoss()

    for idx in range(1, n_iters + 1):
        src_tensor, tgt_tensor, seq_lens = tr_pair_tensors[idx - 1]

        loss = train(
            src_lang, tgt_lang, enc, dec, src_tensor, tgt_tensor, seq_lens,
            tgt_sos_index, enc_optim, dec_optim, criterion,
            batch_size=batch_size
        )

        print_loss_total += loss

        if idx % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_losses.append(print_loss_avg)
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (
                time_since(start, idx / n_iters),
                idx,
                idx / n_iters * 100,
                print_loss_avg))

        if idx % plot_every == 0:
            evaluate_randomly(
                src_lang, tgt_lang, enc, dec, tgt_sos_index, 1, idx)

        if idx % lr_update_every == 0:
            enc_scheduler.step(loss)
            dec_scheduler.step(loss)
            print('iter {0}, enc lr: {1}, dec lr: {2}'.format(
                idx,
                enc_scheduler.optimizer.param_groups[0]['lr'],
                enc_scheduler.optimizer.param_groups[0]['lr']
            ))

    return print_losses
