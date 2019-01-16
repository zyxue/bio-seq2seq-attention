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
        seq_len, batch_size, _ = inputs.shape

        # https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding
        embedded = self.embedding(inputs).view(seq_len, batch_size, -1)

        # https://pytorch.org/docs/stable/nn.html#torch.nn.GRU
        output, hidden = self.gru(embedded, hidden)

        return output, hidden

    def init_hidden(self, batch_size, device):
        directions = 2 if self.bidirectional else 1
        return torch.zeros(
            self.num_layers * directions,
            batch_size,
            self.hidden_size,
            device=device
        )
