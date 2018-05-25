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

from utils import timeSince


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: {0}'.format(DEVICE))

MAX_LENGTH = 500


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1,
                 max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0),
            encoder_outputs.unsqueeze(0)
        )

        out = torch.cat((embedded[0], attn_applied[0]), 1)
        out = self.attn_combine(out).unsqueeze(0)

        out = F.relu(out)
        out, hidden = self.gru(out, hidden)

        output = F.log_softmax(self.out(out[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


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

    # input_length = src_tensor.size(0)
    # target_length = tgt_tensor.size(0)

    # generate encoder outputs for src sequence. 
    # outs indicates an array, out indicates a sinlge step output
    enc_hid = enc.initHidden()
    enc_outs = torch.zeros(MAX_LENGTH, enc.hidden_size, device=DEVICE)
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
               print_every=1000, plot_every=100, learning_rate=0.01):
    print('training for {0} steps'.format(n_iters))
    print('collect loss for plotting per {0} steps'.format(plot_every))

    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    enc_optim = optim.SGD(enc.parameters(), lr=learning_rate)
    dec_optim = optim.SGD(dec.parameters(), lr=learning_rate)

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
                timeSince(start, idx / n_iters),
                idx,
                idx / n_iters * 100,
                print_loss_avg))
            evaluate_randomly(src_lang, tgt_lang, enc, dec, tgt_sos_index, 1)

        if idx % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    # showPlot(plot_losses)


def evaluate(src_lang, tgt_lang, enc, dec, tgt_sos_index,
             src_seq, seq_len, max_length=MAX_LENGTH):
    with torch.no_grad():
        src_tensor = tensor_from_sentence(src_lang, src_seq)

        enc_hid = enc.initHidden()
        enc_outs = torch.zeros(max_length, enc.hidden_size, device=DEVICE)

        for ei in range(seq_len):
            enc_out, enc_hid = enc(src_tensor[ei], enc_hid)
            enc_outs[ei] += enc_out[0, 0]

        dec_in = torch.tensor([[tgt_sos_index]], device=DEVICE)
        dec_hid = enc_hid
        dec_outs = []
        dec_attns = torch.zeros(max_length, max_length)
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
        acc = (np.array(list(tgt_seq)) == np.array(prd_tokens)).sum() / seq_len
        print('acc: {0}'.format(acc))


if __name__ == "__main__":
    # data_dir = '../tf_nmt/nmt_aa_data/Heffernan_2017_SPIDER3/tr_va_te/'
    # data_file = os.path.join(data_dir, 'tr.seq2sst3.pkl')
    data_file = sys.argv[1]

    print('reading data from {0}'.format(os.path.abspath(data_file)))
    with open(data_file, 'rb') as inf:
        src_lang, tgt_lang, pairs = pickle.load(inf)

    print('filter out seqs longer than {0}'.format(MAX_LENGTH))
    pairs = [_ for _ in pairs if _[-1] <= MAX_LENGTH]

    print('data loaded...')
    print('training on {0} seqs'.format(len(pairs)))

    sos_symbol = '^'            # symbol for start of a seq
    tgt_sos_index = src_lang.word2index['^']

    hidden_size = 256
    enc = EncoderRNN(src_lang.n_words, hidden_size)
    enc = enc.to(DEVICE)

    dec = AttnDecoderRNN(
        hidden_size, tgt_lang.n_words, dropout_p=0.1)
    dec.to(DEVICE)

    trainIters(src_lang, tgt_lang, enc, dec, tgt_sos_index,
               n_iters=75000, print_every=10)
