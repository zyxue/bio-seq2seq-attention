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
    def __init__(self, input_size, embedding_size, hidden_size,
                 num_layers=1, bidirectional=False, batch_size=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_size = batch_size

        self.embedding = nn.Embedding(input_size, embedding_size)

        self.gru = nn.GRU(embedding_size, hidden_size, num_layers,
                          bidirectional=bidirectional)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        directions = 2 if self.bidirectional else 1
        return torch.zeros(
            self.num_layers * directions,
            self.batch_size,
            self.hidden_size,
            device=DEVICE
        )


class AttnDecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(
            num_embeddings=output_size,
            embedding_dim=embedding_size
        )
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.attn = nn.Linear(hidden_size, hidden_size)
        # hc: [hidden, context]
        self.Whc = nn.Linear(hidden_size * 2, hidden_size)
        # s: softmax
        self.Ws = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        gru_out, hidden = self.gru(embedded, hidden)

        attn_prod = torch.mm(self.attn(hidden)[0], encoder_outputs.t())
        attn_weights = F.softmax(attn_prod, dim=1)
        context = torch.mm(attn_weights, encoder_outputs)

        # hc: [hidden: context]
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


def train(src_lang, tgt_lang, enc, dec, src_tensor, tgt_tensor, seq_len,
          tgt_sos_index, enc_optim, dec_optim,
          criterion, teacher_forcing_ratio=0.5):
    enc_optim.zero_grad()
    dec_optim.zero_grad()

    # encoding source sequence
    enc_hid = enc.initHidden()
    directions = 2 if enc.bidirectional else 1
    enc_outs = torch.zeros(seq_len, enc.hidden_size * directions, device=DEVICE)
    for ei in range(seq_len):
        enc_out, enc_hid = enc(src_tensor[ei], enc_hid)
        enc_outs[ei] = enc_out[0, 0]

    # take the hidden state from the last step in the encoder, continue in the
    # decoder
    dec_hid = enc_hid
    # init the first input for the decoder
    dec_in = torch.tensor([[tgt_sos_index]], device=DEVICE)

    # decide to use teacher forcing or not
    use_tf = True if random.random() < teacher_forcing_ratio else False

    loss = 0
    if use_tf:
        for di in range(seq_len):
            dec_out, dec_hid, dec_attn = dec(dec_in, dec_hid, enc_outs)
            loss += criterion(dec_out, tgt_tensor[di])
            dec_in = tgt_tensor[di]
    else:
        for di in range(seq_len):
            dec_out, dec_hid, dec_attn = dec(dec_in, dec_hid, enc_outs)
            topv, topi = dec_out.topk(1)
            # Returns a new Tensor, detached from the current graph. The result
            # will never require gradient.
            # https://pytorch.org/docs/stable/autograd.html#torch.Tensor.detach
            dec_in = topi.squeeze().detach()
            loss += criterion(dec_out, tgt_tensor[di])

    loss.backward()

    enc_optim.step()
    dec_optim.step()

    return loss.item() / seq_len


def trainIters(src_lang, tgt_lang, enc, dec, tgt_sos_index, n_iters,
               print_every=1000, plot_every=100, learning_rate=0.001):
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
        tsr = tensors_from_pair(src_lang, tgt_lang, random.choice(pairs))
        tr_pair_tensors.append(tsr)

    criterion = nn.NLLLoss()

    for idx in range(1, n_iters + 1):
        src_tensor, tgt_tensor, seq_len = tr_pair_tensors[idx - 1]

        loss = train(
            src_lang, tgt_lang, enc, dec, src_tensor, tgt_tensor, seq_len,
            tgt_sos_index, enc_optim, dec_optim, criterion
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
        src_tensor = tensor_from_sentence(src_lang, src_seq)

        enc_hid = enc.initHidden()
        enc_outs = torch.zeros(seq_len, enc.hidden_size, device=DEVICE)

        for ei in range(seq_len):
            enc_out, enc_hid = enc(src_tensor[ei], enc_hid)
            enc_outs[ei] += enc_out[0, 0]

        dec_in = torch.tensor([[tgt_sos_index]], device=DEVICE)
        dec_hid = enc_hid
        dec_outs = []
        dec_attns = torch.zeros(seq_len, seq_len)
        for di in range(seq_len):
            dec_out, dec_hid, dec_attn = dec(dec_in, dec_hid, enc_outs)
            dec_attns[di] = dec_attn.data
            topv, topi = dec_out.data.topk(1)
            dec_outs.append(tgt_lang.index2word[topi.item()])

            dec_in = topi.squeeze().detach()

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
        bidirectional=False
    )
    enc = enc.to(DEVICE)

    dec = AttnDecoderRNN(
        args.embedding_size,
        args.hidden_size,
        tgt_lang.n_words,
        dropout_p=0.1
    )
    dec.to(DEVICE)

    trainIters(src_lang, tgt_lang, enc, dec, tgt_sos_index,
               n_iters=args.num_iters, print_every=args.print_every)
