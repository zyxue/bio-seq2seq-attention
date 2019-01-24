import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPDecoder(nn.Module):
    def __init__(self, language, hidden_size, num_layers):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = language.num_tokens

        self.W_hidden_list = [
            nn.Linear(self.hidden_size, self.hidden_size) for _ in num_layers
        ]
        self.W_out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input):
        """input are the output of RNN outputs per time step"""
        out = None
        for W in self.W_hidden_list:
            if out is None:
                out = W(input)
            else:
                out = W(out)
            out = torch.relu(out)
        out = self.W_out(out)
        out = F.log_softmax(out)
        return out


class AttnDecoderRNN(nn.Module):
    def __init__(self, language, embedding_dim, hidden_size, num_layers,
                 dropout_p=0.1):
        """
        :param lang: language
        """
        super(AttnDecoderRNN, self).__init__()

        # keep the language to be decoded for later convenience
        self.language = language

        self.embedding_size = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = language.num_tokens
        self.dropout_p = dropout_p

        # construct network
        self.embedding = nn.Embedding(num_embeddings=language.num_tokens,
                                      embedding_dim=embedding_dim)

        self.dropout = nn.Dropout(self.dropout_p)

        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers)

        # mat: matrix, i.e. weights
        self.mat_attn = nn.Linear(hidden_size, hidden_size)

        # ctx: [hidden, context]
        self.mat_ctx = nn.Linear(hidden_size * 2, hidden_size)

        # weight before log_softmax
        self.mat_stm = nn.Linear(hidden_size, self.output_size)

    def forward(self, inputs, hidden, encoder_out):
        """Shapes:

        Sd: decoder sequence length, it should be one in the decoder since the
        output sequence is decoded one step at a time
        L: sequence length
        B: batch size
        H: hidden size (may or may not times 2 depending on whether encoder is birdirectional, D will be omitted for clarity)
        Y: num_hidden_layers
        C: num_tokens/classes

        input: B x 1
        hidden: Y x B x H
        encoder_outputs: L x B x H
        """

        L, B = inputs.shape

        # self.embedding shape: C x E
        # inputs.shape: L x B
        # emb.shape: L x B x E
        emb = self.embedding(inputs)

        # gru_out.shape: 1 x B x H, 1 because decoder always decodes one step at a time
        # gru_hid.shape: Y x B x H
        gru_out, gru_hid = self.gru(emb, hidden)

        # TODO: think through drop later by reading papers
        # emb = self.dropout(emb)

        # seq_len = encoder_outputs.shape[0]
        # layer_x_direc_size, batch_size, hidden_size = hidden.shape

        # prepare out for attention calculation, shape doesn't change
        gru_out_attn = self.mat_attn(gru_out)

        # for each batch, dot product gru_out with every element in
        # encoder_out, applying broadcasting
        # encoder_out.shape: L x B x H
        # gru_out.shape: 1 x B x H
        # attn_prod.shape: L x B
        attn_prod = (gru_out_attn * encoder_out).sum(dim=2)

        # TODO: confirm what is attention smoothing.
        # https://arxiv.org/pdf/1707.07167.pdf

        # attention weights: use sigmoid instead of softmax because when sequence is long,
        # probabilities tend to become tiny
        attn_weig = torch.sigmoid(attn_prod)

        # ctx.shape: L x B
        ctx = (attn_weig.unsqueeze(dim=2) * encoder_out).sum(dim=0)

        # cat ctx and state of hidden layer
        ctx_cat = torch.cat([ctx, gru_hid[-1]], dim=1)

        # TODO: alternatively, maybe try this version
        # gru_out if of seq_len, so [0] is equivalent to squeeze the first dim
        # ctx_cat = torch.cat([ctx, gru_out[0]], dim=1)
        ctx_out = torch.tanh(self.mat_ctx(ctx_cat))

        out = F.log_softmax(self.mat_stm(ctx_out), dim=1)
        return out, gru_hid, attn_weig
