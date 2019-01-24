import torch


def decode(decoder, data_batch, encoder_out, loss_func):
    seq0s, seq1s, seq_lens = data_batch  # all in indices
    padded_seq_len, batch_size = seq0s.shape

    loss = torch.zeros(batch_size, device=seq0s.device)

    for i in range(padded_seq_len):
        inp = encoder_out[i]
        out, hid, attn = decoder(inp)

        # out.shape: B x C
        # seq1s.shape: L x B
        _l = loss_func(out, seq1s[i])  # padded tokens will have 0 loss
        loss += _l

        inp = torch.argmax(out, dim=1).unsqueeze(dim=0)

    mean_loss = torch.mean(loss / seq_lens.float())
    return mean_loss
