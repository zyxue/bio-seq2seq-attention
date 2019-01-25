import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPDecoder(nn.Module):
    def __init__(self, language, hidden_size, num_layers, device=None):
        super(MLPDecoder, self).__init__()
        self.language = language

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = language.num_tokens

        self.W_hidden_list = [
            nn.Linear(self.hidden_size, self.hidden_size) for _ in range(num_layers)
        ]
        self.W_out = nn.Linear(self.hidden_size, self.output_size)

        if device is not None:
            self.to(device)

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
        out = F.log_softmax(out, dim=1)
        return out
