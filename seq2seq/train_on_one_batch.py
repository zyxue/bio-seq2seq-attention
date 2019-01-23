import random

import torch
import numpy as np


def encode(encoder, data_batch):
    """
    run data through encoder and return the encoded tensors

    data_batch is a tuple of (seq0s, seq1s, seq_lens).
    seq0s and seq1s should be in indices
    """
    seq0s, _, _ = data_batch
    # seq0s.shape: L x B
    _, B = seq0s.shape

    hid = encoder.init_hidden(B, device=seq0s.device)
    out, hid = encoder(seq0s, hid)

    return out, hid


def init_decoder_input(lang, batch_size, device):
    """
    initialize decoder input, i.e. prepare a batch of beginning tokens

    for batch_size of 2, the return value will be like
    tensor([[0, 0]])

    L x B and L=1
    """
    idx = lang.token2index[lang.beg_token]
    # repeat method in torch not found
    # reshape following convention: 
    arr = np.repeat([[idx]], batch_size, axis=1)
    return torch.tensor(arr, device=device)


def use_tf(ratio):
    """use teacher forcing or not"""
    return True if random.random() < ratio else False


def decode(decoder, data_batch, encoder_out, encoder_hidden, loss_func,
           tf_ratio):
    """
    :param tf_ratio: teacher enforcing ratio
    """
    seq0s, seq1s, seq_lens = data_batch  # all in indices
    padded_seq_len, batch_size = seq0s.shape

    # init the first output token for the decoder, L x B and L=1
    inp = init_decoder_input(decoder.language, batch_size, device=seq0s.device)

    # inherit the hidden state from the last step output from the encoder
    hid = encoder_hidden

    tf_ratio = 0
    loss = torch.zeros(batch_size, device=seq0s.device)
    for i in range(padded_seq_len):
        out, hid, attn = decoder(inp, hid, encoder_out)

        # out.shape: B x C
        # seq1s.shape: L x B
        _l = loss_func(out, seq1s[i])  # padded tokens will have 0 loss
        loss += _l

        if use_tf(tf_ratio):
            inp = seq1s[i]
        else:
            inp = torch.argmax(out, dim=1).unsqueeze(dim=0)

    mean_loss = torch.mean(loss / seq_lens.float())
    return mean_loss


def train_on_one_batch(encoder, decoder, encoder_optim, decoder_optim,
                       data_batch, loss_func, tf_ratio):
    """
    :param tf_ratio: teacher_forcing ratio
    """
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    enc_out, enc_hid = encode(encoder, data_batch)

    loss = decode(decoder, data_batch, enc_out, enc_hid, loss_func, tf_ratio)

    loss.backward()

    encoder_optim.step()
    decoder_optim.step()

    return loss.item()
