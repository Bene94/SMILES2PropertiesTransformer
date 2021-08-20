import math
import time

import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.modules import activation


class TransformerModel(nn.Module):

    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(config.embed_size, config.dropout, max_len=config.block_size)
        encoder_layers = TransformerEncoderLayer(config.embed_size, config.num_heads, dim_feedforward = config.hidden_size, dropout = config.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, config.num_layers)
        self.encoder = nn.Embedding(config.vocab_size, config.embed_size, padding_idx = config.padding_idx)
        self.ninp = config.embed_size

        self.dense_list = nn.ModuleList([nn.Linear(config.embed_size, config.embed_size) for _ in range(config.n_dense)])
        self.dropout_list = nn.ModuleList([nn.Dropout(config.dense_dropout) for _ in range(config.n_dense)])
        
        self.decoder = nn.Linear(config.embed_size, 1)
        self.pool = nn.MaxPool1d(kernel_size = config.block_size, stride = config.block_size)

        self.init_weights(config)

    def init_weights(self, config):
        self.encoder.weight.data = nn.init.xavier_normal_(self.encoder.weight.data)
        
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.normal_(0, math.sqrt(2/config.embed_size))

    def forward(self, src, src_key_padding_mask):
        src = src.type(torch.cuda.IntTensor)
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.pool(output.permute(0,2,1)).permute(0,2,1)

        for i in range(len(self.dense_list)):
            output = self.dense_list[i](output)
            output = F.relu(output)
            output = self.dropout_list[i](output)

        output = self.decoder(output)

        return output
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)