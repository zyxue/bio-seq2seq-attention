import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnDecoderRNN(nn.Module):
    def __init__(self, language, embedding_size, hidden_size, num_layers,
                 dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()

        # keep the language to be decoded for later convenience
        self.language = language
        output_size = language.num_tokens

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
        L: sequence length
        B: batch size
        H: hidden size

        input: B x 1
        hidden: 1 x B x H
        encoder_outputs: L x B x H
        """

        batch_size = input.shape[0]
        # 1 means one step: decoder always decodes one step at a time
        embedded = self.embedding(input).view(1, batch_size, -1)
        embedded = self.dropout(embedded)

        seq_len = encoder_outputs.shape[0]
        layer_x_direc_size, batch_size, hidden_size = hidden.shape

        gru_out, hidden = self.gru(embedded, hidden)

        # L x B
        attn_prod = torch.mul(self.attn(gru_out), encoder_outputs).sum(dim=2)

        # attn_weights = F.softmax(attn_prod, dim=0)
        # attention smoothing. https://arxiv.org/pdf/1707.07167.pdf
        attn_weights = F.sigmoid(attn_prod)

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
