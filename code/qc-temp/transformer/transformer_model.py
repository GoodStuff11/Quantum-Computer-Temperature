import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
import time
import psutil

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


class EncoderOnlyTransformerModel(nn.Module):
    def __init__(
        self,
        atom_grid_shape: tuple,
        spin_states: int,
        nhead: int,
        nlayers: int = 6,
        embedding_size: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.5,
        n_phys_params: int = 1,
    ):
        """Encoder-only transformer

        Args:
            atom_grid_shape (tuple): Allows only for grids of atoms. For 1d grid, put (x_length,) and
                2d grids are (x_length, y_length)
            spin_states (int): Number of states in system, such as 2 for a up, down spin state
            embedding_size (int): Dimension of embedding. (Encodes with one-hot)
            nhead (int): Number of heads
            dim_feedforward (int): the dimension of the feedforward network model 
            nlayers (int): Number of encoder layers
            dropout (float, optional): Dropout fraction. Defaults to 0.5.
            n_phys_params (int, optional): Number of physical parameters to append to start of input. Defaults to 1.
        """
        super().__init__()
        self.start_iteration = 0
        
        
        self.model_type = 'Transformer'
        self.spin_states = spin_states
        self.embedding_size = embedding_size
        self.natoms = math.prod(atom_grid_shape)
        self.n_phys_params = n_phys_params  # num parameters to put at start of input: [beta]
        self.hyperparameters = {
            "atom_grid_shape": atom_grid_shape,
            "spin_states": spin_states,
            "nhead": nhead,
            "nlayers": nlayers,
            "embedding_size": embedding_size,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "n_phys_params": n_phys_params,
        }

        self.encoder = nn.Linear(self.n_phys_params + spin_states, embedding_size)
        self.pos_encoder = PositionalEncoding(self.embedding_size, dropout)
        encoder_layers = TransformerEncoderLayer(self.embedding_size, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(self.embedding_size, self.spin_states)

        self.optim = TransformerOptimizer(self)

    def embedding(self, spins: Tensor, phys_params: Tensor) -> Tensor:
        """
        encoded:
            0 1 0 0 -> [[beta 0 0],
                        [0 1 0],
                        [0 0 1],
                        [0 1 0],
                        [0 1 0]]
            (self.phys_params + natoms, self.phys_params + spin_states)


        hello you -> [[1 2 3 5 1],
                      [2 6 3 2 9]]


        Args:
            spins (Tensor): tensor of size (natoms, batchsize) composed of 0s and 1s
            phys_params (Tensor): physical parameters, such as beta in a tensor of size (n_phys_params,)

        Returns:
            Tensor: Tensor of size (n_phys_params + natoms, batch_size, n_phys_params + spin_states) which
                embeds spins and physical parameters into a single tensor.
        """
        _, batch_size = spins.shape
        init = torch.zeros((self.n_phys_params + self.natoms, batch_size, self.n_phys_params + self.spin_states))
        init[: self.n_phys_params, :, : self.n_phys_params] = phys_params.diag_embed(dim1=0, dim2=2)
        init[self.n_phys_params :, :, self.n_phys_params :] += F.one_hot(spins, num_classes=self.spin_states)
        
        return init

    def forward(self, spins: Tensor, phys_params: Tensor) -> Tensor:
        """ "

        Arguments:
            spins: Tensor, shape ``[natoms, batch_size]``
            phys_params: Tensor, shape ``[self.n_phys_params, batch_size]``

        Returns:
            output Tensor of shape ``[natoms, batch_size, self.spin_states]``
        """
        # `print(`"Input: ", spins.T)
        src = self.encoder(self.embedding(spins, phys_params.T)) * math.sqrt(self.embedding_size)
        src = self.pos_encoder(src)
        src_mask = generate_square_subsequent_mask(len(src), diag=1)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        output = F.log_softmax(output, dim=-1)
        return output[self.n_phys_params - 1: -1]

    def start_training(self, dataloader, checkpoint_path: str = './checkpoints/model.pt', data_file:str='./checkpoints/data1.csv', log_interval: int=100):
        """_summary_

        Args:
            dataloader (_type_): Should be an iterator that outputs (data, beta) pairs where
                data has size [natoms, batchsize] and beta has size [n_phys_params, batchsize]
            checkpoint_path (str, optional): _description_. Defaults to './checkpoints/model.pt'.
        """
        total_loss = 0.0

        self.train()
        start_time = time.time()
        for i, (spins, parameters) in enumerate(dataloader):
            log_p = self(spins, parameters)[:-1].view(-1, 2)
            target = spins[:-1].view(-1)

            loss = self.optim.step(log_p, target)

            total_loss += loss
            if i % log_interval == 0 and i > 0:
                lr = self.optim.scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                with open(data_file, 'a+') as f:
                    display_output = f'epoch {dataloader.current_epoch:3d} | {i%dataloader.nbatches:5d}/{dataloader.nbatches:d} batches | '\
                                     f'lr {lr:02.6f} | ms/batch {ms_per_batch:5.2f} | '\
                                     f'loss {cur_loss:5.4f} | memory {psutil.Process(os.getpid()).memory_percent()}'
                    f.write(display_output + '\n')
                print(display_output)
                total_loss = 0
                
                self.save(checkpoint_path)

                start_time = time.time()

    def save(self, checkpoint_path: str = './checkpoints/model.pt'):
        torch.save(
            {
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'hyperparameters': self.hyperparameters,
            },
            checkpoint_path,
        )

    @classmethod
    def load(cls, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model = cls(**checkpoint['hyperparameters'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        return model

def generate_square_subsequent_mask(sz: int, diag:int=1) -> Tensor:
    """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=diag)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)[:,:(-1 if d_model%2 else None)]
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)

class TQSPositionalEncoding2D(nn.Module):
    """
        A mixture of learnable and fixed positional encoding
        the first param_dim inputs have a learnable pe (parameter part)
        the rest have a sinusoidal pe (physical dimensions)
    """

    def __init__(self, d_model: int, param_dim: int, system_size: tuple, dropout: float=0):
        super(TQSPositionalEncoding2D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        system_size = np.array(system_size).reshape(-1)
        self.system_size = system_size
        self.param_embedding = nn.Parameter(
            torch.empty(param_dim, 1, d_model).normal_(std=0.02))  # (param_dim, 1, d_model)

        assert len(system_size) == 2

        x, y = system_size
        channels = int(np.ceil(d_model / 4) * 2)
        div_term = torch.exp(torch.arange(0, channels, 2, dtype=torch.get_default_dtype()) * (
                -math.log(10000.0) / channels))  # channels/2
        pos_x = torch.arange(x, dtype=div_term.dtype).unsqueeze(1)  # (nx, 1)
        pos_y = torch.arange(y, dtype=div_term.dtype).unsqueeze(1)  # (ny, 1)
        sin_inp_x = pos_x * div_term  # (nx, channels/2)
        sin_inp_y = pos_y * div_term  # (ny, channels/2)
        emb_x = torch.zeros(x, channels)
        emb_y = torch.zeros(y, channels)
        emb_x[:, 0::2] = sin_inp_x.sin()
        emb_x[:, 1::2] = sin_inp_x.cos()
        emb_y[:, 0::2] = sin_inp_y.sin()
        emb_y[:, 1::2] = sin_inp_y.cos()
        pe = torch.zeros((x, y, 2 * channels))
        pe[:, :, :channels] = emb_x.unsqueeze(1)
        pe[:, :, channels:] = emb_y  # (x, y, 2*channels)
        pe = pe[:, :, :d_model]  # (x, y, d_model)
        pe = pe.unsqueeze(2)  # (x, y, 1, d_model)

        self.pe = pe  # (x, y, 1, d_model)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
            system_size: the size of the system. Default: None, uses max_system_size
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        pe = self.pe[:self.system_size[0], :self.system_size[1]].reshape(-1, 1, self.d_model)
        pe = torch.cat([self.param_embedding, pe], dim=0)  # (param_dim+n, 1, d_model)
        x = x + pe[:x.size(0)]
        return self.dropout(x)


class TransformerOptimizer:
    def __init__(self, model) -> None:
        self.loss_fn = nn.NLLLoss()
        self.optim = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim, lambda step: 0.001, #self.lr_schedule(step, model.embedding_size)
        )

    @staticmethod
    def lr_schedule(step, model_size, factor=5.0, warmup=4000):
        step = max(step, 1)
        # using the lr schedule from the paper: Attention is all you need
        return factor * (model_size ** (-0.5) * min(step ** (-0.75), step * warmup ** (-1.75)))

    def state_dict(self):
        return self.optim.state_dict(), self.scheduler.state_dict()

    def load_state_dict(self, state_dict: tuple):
        self.optim.load_state_dict(state_dict[0])
        self.scheduler.load_state_dict(state_dict[1])
        
    def step(self, log_p, target):
        # print("Output: ",log_p.exp())
        # print("Target: ", target)
        loss = self.loss_fn(log_p, target)
        # print('-'*10)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.scheduler.step()

        return loss.item()

