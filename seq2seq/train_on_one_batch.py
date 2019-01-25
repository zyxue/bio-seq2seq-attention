import torch
from torch.nn.functional import log_softmax


def encode(encoder, data_batch):
    """
    run data through encoder and return the encoded tensors

    data_batch is a tuple of (seq0s, seq1s, seq_lens).
    seq0s and seq1s should be in indices
    """
    seq0s, _, _ = data_batch
    out, hid = encoder(seq0s)
    return out, hid


def decode(decoder, data_batch, encoder_out, loss_func):
    """decode one base at a time for all sequences within one batch"""

    seq0s, seq1s, seq_lens = data_batch

    seq_len, batch_size = seq0s.shape

    loss = torch.zeros(batch_size, device=seq0s.device)

    for i in range(seq_len):
        inp = encoder_out[i]
        out = decoder(inp)
        out = log_softmax(out, dim=1)

        # out.shape: B x C
        # seq1s.shape: L x B
        _l = loss_func(out, seq1s[i])  # padded tokens will have 0 loss
        loss += _l

    mean_loss = torch.mean(loss / seq_lens.float())
    return mean_loss


def train_on_one_batch(encoder, decoder, encoder_optim, decoder_optim,
                       data_batch, loss_func):
    """
    :param tf_ratio: teacher_forcing ratio
    """
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    enc_out, enc_hid = encode(encoder, data_batch)

    loss = decode(decoder, data_batch, enc_out, loss_func)

    loss.backward()

    encoder_optim.step()
    decoder_optim.step()

    return loss.item()
