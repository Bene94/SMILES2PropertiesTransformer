# test transformer

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import time

from torch.nn.modules import activation


class TransformerModel(nn.Module):

    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(config.embed_size, config.dropout)
        encoder_layers = TransformerEncoderLayer(config.embed_size, config.num_heads, config.hidden_size, config.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, config.num_layers)
        self.encoder = nn.Embedding(config.ntokens, config.embed_size, padding_idx = config.padding_idx)
        self.ninp = config.embed_size
        self.dense = nn.Linear(config.embed_size, config.embed_size)
        self.decoder = nn.Linear(config.embed_size, 1)
        self.pool = nn.MaxPool1d(kernel_size = 256, stride = 256)

        self.init_weights()


    def init_weights(self):
        initrange = 1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_key_padding_mask):
        src = src.type(torch.cuda.IntTensor)
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.pool(output.permute(0,2,1)).permute(0,2,1)
        output = F.relu(self.dense(output))
        output = self.decoder(output)
        return output

    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
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

def train(model, criterion, optimizer, train_dataloader, scheduler, epoch, wandb):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    for i, batch in enumerate(train_dataloader):
        data, targets = batch[0], batch[1]
        optimizer.zero_grad()
        src_padding_mask = (data == wandb.config.padding_idx).transpose(0, 1)
        output = model(data, src_key_padding_mask = src_padding_mask)
        loss = criterion(output, targets)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # gradient clipping this is questionable
        optimizer.step()

        total_loss += loss.item()
        log_interval = 10
      
        wandb.log({"train_loss": loss.item()})
        wandb.log({"grad_norm": torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)})
        wandb.log({"lr": optimizer.param_groups[0]['lr']})
        wandb.log({"epoch": epoch})
        wandb.log({"batch_time": time.time() - start_time})
      
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f}'.format(
                    epoch, i, len(train_dataloader), scheduler.get_last_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, val_dataloader, criterion, config):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(len(val_dataloader)):
            data, targets = next(iter(val_dataloader))
            data = data.type(torch.IntTensor).to(config.device)
            targets = targets.type(torch.FloatTensor).to(config.device)
            targets = targets.view((targets.shape[0],1,1))
            src_key_padding_mask = data.eq(35)
            src_key_padding_mask = src_key_padding_mask.permute(1,0)
            output = eval_model(data, src_key_padding_mask = src_key_padding_mask)
            total_loss += criterion(output, targets).item()
    return total_loss / (len(val_dataloader))

