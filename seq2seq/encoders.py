import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, language, embedding_dim, hidden_size, num_layers,
                 bidirectional=False):
        super(EncoderRNN, self).__init__()

        # keep the language to be encoded for later convenience
        self.language = language

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(language.num_tokens, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers,
                          bidirectional=bidirectional)

    def forward(self, inputs, hidden):
        """
        size variables:
        L: seq_len
        B: batch_size
        E: embedding_dim
        H: hidden_size
        D: num_directions
        Y: num_hidden_layers
        C: num_tokens/classes
        """
        Y = self.num_layers
        H = self.hidden_size
        D = 2 if self.bidirectional else 1

        L, B = inputs.shape

        # self.embedding shape: C x E
        # inputs.shape: L x B
        # emb.shape: L x B x E
        emb = self.embedding(inputs)

        # out.shape: L x B x (D * H)
        # hid.shape: (D * Y) x B x H
        out, hid = self.gru(emb, hidden)

        if not self.bidirectional:
            return hid

        # concat hidden neurons from two RNN per layer
        hid = hid.view(Y, D, B, H)
        h0 = hid[:, 0, :, :]
        h1 = hid[:, 1, :, :]
        # hid_bi.shape: Y x B x (D * H)
        hid_bi = torch.cat([h0, h1], dim=2)
        return out, hid_bi

    def init_hidden(self, batch_size, device):
        directions = 2 if self.bidirectional else 1
        return torch.zeros(self.num_layers * directions,
                           batch_size,
                           self.hidden_size,
                           device=device)
