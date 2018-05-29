import os
import sys
import random
import pickle
import time

import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import utils as U


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: {0}'.format(DEVICE))


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers,
                 bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers,
                          bidirectional=bidirectional)

    def forward(self, inputs, hidden):
        seq_len, batch_size, _ = inputs.shape
        # https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding
        embedded = self.embedding(inputs).view(seq_len, batch_size, -1)
        # https://pytorch.org/docs/stable/nn.html#torch.nn.GRU
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        directions = 2 if self.bidirectional else 1
        return torch.zeros(
            self.num_layers * directions,
            batch_size,
            self.hidden_size,
            device=DEVICE
        )


class AttnDecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, output_size,
                 dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(
            num_embeddings=output_size,
            embedding_dim=embedding_size
        )
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers)
        self.attn = nn.Linear(hidden_size, hidden_size)
        # hc: [hidden, context]
        self.Whc = nn.Linear(hidden_size * 2, hidden_size)
        # s: softmax
        self.Ws = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        """Shapes:

        Sd: decoder sequence length, it should be one in the decoder since the
        output sequence is decoded one step at a time
        S: sequence length
        B: batch size
        H: hidden size

        input: B x 1
        hidden: 1 x B x H
        encoder_outputs: S x B x H
        """

        batch_size = input.shape[0]
        # 1 means one step: decoder always decodes one step at a time
        embedded = self.embedding(input).view(1, batch_size, -1)
        embedded = self.dropout(embedded)

        seq_len = encoder_outputs.shape[0]
        layer_x_direc_size, batch_size, hidden_size = hidden.shape

        gru_out, hidden = self.gru(embedded, hidden)

        # S x B
        attn_prod = torch.mul(gru_out, encoder_outputs).sum(dim=2)

        attn_weights = F.softmax(attn_prod, dim=0)
        # B x H: weighted average
        context = torch.mul(
            # .view: make attn_weights 3D tensor to make it multiplicable
            attn_weights.view(seq_len, batch_size, 1),
            encoder_outputs
        ).sum(dim=0)

        hc = torch.cat([hidden[0], context], dim=1)
        out_hc = F.tanh(self.Whc(hc))
        output = F.log_softmax(self.Ws(out_hc), dim=1)

        return output, hidden, attn_weights


def tensor_from_sentence(lang, sentence):
    indexes = [lang.word2index[i] for i in sentence]
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)


def tensors_from_pair(src_lang, tgt_lang, pair):
    src_tensor = tensor_from_sentence(src_lang, pair[0])
    tgt_tensor = tensor_from_sentence(tgt_lang, pair[1])
    seq_len = pair[2]
    return (src_tensor, tgt_tensor, seq_len)


def train(src_lang, tgt_lang, enc, dec, src_tensor, tgt_tensor, seq_lens,
          tgt_sos_index, enc_optim, dec_optim,
          criterion, teacher_forcing_ratio=0.5, batch_size=1):
    enc_optim.zero_grad()
    dec_optim.zero_grad()

    # encoding source sequence
    enc_hid = enc.init_hidden(batch_size)
    directions = 2 if enc.bidirectional else 1
    enc_outs, enc_hid = enc(src_tensor, enc_hid)

    # take the hidden state from the last step in the encoder, continue in the
    # decoder
    dec_hid = enc_hid
    # init the first input for the decoder
    dec_in = torch.tensor([[tgt_sos_index] * batch_size], device=DEVICE).view(-1, 1)

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


def trainIters(src_lang, tgt_lang, enc, dec, tgt_sos_index, n_iters,
               batch_size=1, print_every=1000, plot_every=100, learning_rate=0.001):
    print('training for {0} steps'.format(n_iters))
    print('collect loss for plotting per {0} steps'.format(plot_every))

    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    # enc_optim = optim.SGD(enc.parameters(), lr=learning_rate)
    # dec_optim = optim.SGD(dec.parameters(), lr=learning_rate)
    enc_optim = optim.Adam(enc.parameters(), lr=learning_rate)
    dec_optim = optim.Adam(dec.parameters(), lr=learning_rate)

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
        plot_loss_total += loss

        if idx % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (
                U.time_since(start, idx / n_iters),
                idx,
                idx / n_iters * 100,
                print_loss_avg))
            evaluate_randomly(src_lang, tgt_lang, enc, dec, tgt_sos_index, 1)

        if idx % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    # showPlot(plot_losses)


def evaluate(src_lang, tgt_lang, enc, dec, tgt_sos_index, src_seq, seq_len):
    with torch.no_grad():
        # shape: S X B X 1
        src_tensor = tensor_from_sentence(src_lang, src_seq).view(-1, 1, 1)

        enc_hid = enc.init_hidden(batch_size=1)
        enc_outs, enc_hid = enc(src_tensor, enc_hid)

        dec_in = torch.tensor([[tgt_sos_index]], device=DEVICE).view(-1, 1)
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


def evaluate_randomly(src_lang, tgt_lang, enc, dec, tgt_sos_index, n=10):
    for i in range(n):
        src_seq, tgt_seq, seq_len = random.choice(pairs)
        print('>', src_seq)
        print('=', tgt_seq)
        prd_tokens, attns = evaluate(
            src_lang, tgt_lang, enc, dec, tgt_sos_index, src_seq, seq_len)
        prd_seq = ''.join(prd_tokens)
        print('<', prd_seq)
        acc = U.calc_accuracy(tgt_seq, prd_seq)
        print('acc: {0}'.format(acc))


if __name__ == "__main__":
    args = U.parse_args()

    data_file = args.input

    for i in [
            'embedding_size',
            'hidden_size',
            'batch_size',
            'num_layers',
            'bidirectional',
    ]:
        print('{0}:\t{1}'.format(i, getattr(args, i)))

    print('reading data from {0}'.format(os.path.abspath(data_file)))
    with open(data_file, 'rb') as inf:
        src_lang, tgt_lang, pairs = pickle.load(inf)

    # print('filter out seqs longer than {0}'.format(MAX_LENGTH))
    # pairs = [_ for _ in pairs if _[-1] <= MAX_LENGTH]

    print('data loaded...')
    print('training on {0} seqs'.format(len(pairs)))

    sos_symbol = '^'            # symbol for start of a seq
    tgt_sos_index = src_lang.word2index['^']

    enc = EncoderRNN(
        src_lang.n_words,
        args.embedding_size,
        args.hidden_size,
        args.num_layers,
        args.bidirectional,
    )
    enc = enc.to(DEVICE)

    dec = AttnDecoderRNN(
        args.embedding_size,
        args.hidden_size,
        args.num_layers,
        tgt_lang.n_words,
        dropout_p=0.1
    )
    dec.to(DEVICE)

    trainIters(src_lang, tgt_lang, enc, dec, tgt_sos_index,
               n_iters=args.num_iters, print_every=args.print_every,
               batch_size=args.batch_size
    )
