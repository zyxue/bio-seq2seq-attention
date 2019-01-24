import random
import torch

from seq2seq.plot import plot_attn
# from seq2seq.utils import tensor_from_sentence, get_device


def evaluate(src_lang, tgt_lang, enc, dec, tgt_sos_index, src_seq, seq_len):
    with torch.no_grad():
        # shape: S X B X 1
        src_tensor = tensor_from_sentence(src_lang, src_seq).view(-1, 1, 1)

        enc_hid = enc.init_hidden(batch_size=1)
        enc_outs, enc_hid = enc(src_tensor, enc_hid)

        if enc.bidirectional:
            # as the enc_outs has a 2x factor for hidden size, so reshape hidden to
            # match that
            enc_hid = torch.cat([
                enc_hid[:enc.num_layers, :, :],
                enc_hid[enc.num_layers:, :, :]
            ], dim=2)

        device = get_device()
        dec_in = torch.tensor([[tgt_sos_index]], device=device).view(-1, 1)
        dec_hid = enc_hid
        dec_outs = []
        dec_attns = torch.zeros(seq_len, seq_len)
        for di in range(seq_len):
            dec_out, dec_hid, dec_attn = dec(dec_in, dec_hid, enc_outs)
            dec_attns[di] = dec_attn.view(-1)
            topv, topi = dec_out.data.topk(1)
            dec_outs.append(tgt_lang.index2word[topi.item()])

            dec_in = topi.detach()

        return dec_outs, dec_attns[:di + 1]


def evaluate_randomly(src_lang, tgt_lang, enc, dec, tgt_sos_index,
                      num, iter_idx):
    for i in range(num):
        src_seq, tgt_seq, seq_len = random.choice(pairs)
        print('>', src_seq)
        print('=', tgt_seq)
        prd_tokens, attns = evaluate(
            src_lang, tgt_lang, enc, dec, tgt_sos_index, src_seq, seq_len)
        prd_seq = ''.join(prd_tokens)
        print('<', prd_seq)
        acc = U.calc_accuracy(tgt_seq, prd_seq)
        print('acc: {0}'.format(acc))
        plot_attn(attns, src_seq, prd_seq, acc, iter_idx)
