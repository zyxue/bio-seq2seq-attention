from seq2seq import decode_with_mlp


def encode(encoder, data_batch):
    """
    run data through encoder and return the encoded tensors

    data_batch is a tuple of (seq0s, seq1s, seq_lens).
    seq0s and seq1s should be in indices
    """
    seq0s, _, _ = data_batch
    out, hid = encoder(seq0s)
    return out, hid


def train_on_one_batch(encoder, decoder, encoder_optim, decoder_optim,
                       data_batch, loss_func):
    """
    :param tf_ratio: teacher_forcing ratio
    """
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    enc_out, enc_hid = encode(encoder, data_batch)

    loss = decode_with_mlp.decode(decoder, data_batch, enc_out, loss_func)

    loss.backward()

    encoder_optim.step()
    decoder_optim.step()

    return loss.item()
