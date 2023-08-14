import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchscale.architecture.config import RetNetConfig
from torchscale.architecture.retnet import RetNetDecoder


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.Lx = None
        self.Ly = None

    @property
    def N(self):
        return self.Lx * self.Ly

    def forward(self, x):
        pass

    def sample(self, nsamples):
        pass

    def logp(self, x):
        """_summary_

        Args:
            x (tensor): (batchsize, natoms)

        Returns:
            tensor: (batchsize,)
        """
        logp = self(x)
        return torch.sum(logp * F.one_hot(x, num_classes=2), dim=(-2, -1))


class RNN_1D(Model):
    def __init__(self, Lx, Ly, hidden_size=10, model_type='lstm'):
        super().__init__()
        self.Lx, self.Ly = Lx, Ly
        self.hidden_size = hidden_size
        if model_type == "lstm":
            model = nn.LSTM
        elif model_type == 'gru':
            model = nn.GRU
        elif model_type == 'rnn':
            model = nn.RNN
            
        self.rnn = model(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.to(self.device)

    @torch.jit.export
    def forward(self, x):
        x = torch.cat((torch.zeros((x.shape[0], 1), device=self.device), x[:, :-1]), dim=1).type(torch.int64)
        x = F.one_hot(x, num_classes=2).type(torch.float32)
        output, _ = self.rnn(x)
        output = self.linear(output)
        output = self.log_softmax(output)
        return output

    @torch.jit.export
    def sample(self, nsamples):
        h = None
        samples = torch.zeros(
            (nsamples, self.N), device=self.device, dtype=torch.int64
        )  # the first guess is when inputting zero
        for i in range(self.N):
            i = max(i, 1)
            input = F.one_hot(samples[:, i - 1 : i], num_classes=2).type(torch.float32)
            output, h = self.rnn.forward(input, h)
            output = self.linear(output)
            output = F.softmax(output, dim=-1)
            samples[:, i - 1] = torch.multinomial(output.reshape(nsamples, 2), 1).reshape(nsamples)
        return samples


class Transformer(Model):
    def __init__(self):
        super().__init__()

        self.to(self.device)


class RetNet(Model):
    def __init__(
        self, Lx, Ly, decoder_ffn_embed_dim: int = 300, decoder_layers: int = 3, embedding_dim=12, nheads=3
    ):
        super().__init__()
        self.Lx, self.Ly = Lx, Ly
        # vocab size is before embedding. decoder_embed_dim is the number of dimensions in your embedding, ideally less than
        config = RetNetConfig(
            vocab_size=2,
            decoder_embed_dim=embedding_dim,
            decoder_ffn_embed_dim=decoder_ffn_embed_dim,
            decoder_layers=decoder_layers,
            decoder_retention_heads=nheads,
        )
        # adding an embedding that is a higher dimension than the vocabulary so that we can have more than one
        # head. Otherwise, it won't fit propertly
        embedding = nn.Embedding(2, embedding_dim)
        self.retnet = RetNetDecoder(config, embed_tokens=embedding)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # required
        self.to(self.device)

    @torch.jit.export
    def forward(self, x):
        """_summary_

        Args:
            x (tensor): shape (batchsize, N)

        Returns:
            _type_: _description_
        """
        x = torch.cat((torch.zeros((x.shape[0], 1), device=self.device), x[:, :-1]), dim=1).type(torch.int64)
        x, _ = self.retnet(x)
        x = self.log_softmax(x)

        return x

    @torch.jit.export
    def sample(self, nsamples):
        incremental_state = {}
        samples = torch.zeros(
            (nsamples, self.N), device=self.device, dtype=torch.int64
        )  # the first guess is when inputting zero
        for i in range(self.N):
            i = max(i, 1)
            output, _ = self.retnet.forward(samples[:, i - 1 : i], incremental_state=incremental_state)
            output = F.softmax(output, dim=-1)
            samples[:, i - 1] = torch.multinomial(output.reshape(nsamples, 2), 1).reshape(nsamples)
        return samples
