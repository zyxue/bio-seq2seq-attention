from seq2seq import encode
from seq2seq import decode_with_attn
from seq2seq import decode_with_mlp


def train_on_one_batch(encoder, decoder, encoder_optim, decoder_optim,
                       data_batch, loss_func, tf_ratio, architecture):
    """
    :param tf_ratio: teacher_forcing ratio
    """
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    enc_out, enc_hid = encode.encode(encoder, data_batch)

    if architecture == "encoder-decoder":
        loss = decode_with_attn.decode(decoder, data_batch, enc_out, enc_hid,
                                       loss_func, tf_ratio)
    elif architecture == "rnn+mlp":
        loss = decode_with_mlp.decode(decoder, data_batch, enc_out,
                                      loss_func)

    loss.backward()

    encoder_optim.step()
    decoder_optim.step()

    return loss.item()
