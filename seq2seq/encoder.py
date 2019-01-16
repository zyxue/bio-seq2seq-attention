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
        E: embedding size
        H: hidden size
        D: num_directions
        Y: num_hidden_layers
        """
        L, B = inputs.shape

        # shape: L x B x E
        emb = self.embedding(inputs)

        # output shape: L x B x (D * H)
        # hidden shape: (D * Y) x B x H
        out, hidden = self.gru(emb, hidden)

        return out, hidden

    def init_hidden(self, batch_size):
        directions = 2 if self.bidirectional else 1
        return torch.zeros(self.num_layers * directions,
                           batch_size,
                           self.hidden_size)
