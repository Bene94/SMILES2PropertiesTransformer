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
        self.pos_encoder = PositionalEncoding(config.embed_size, config.dropout)
        encoder_layers = TransformerEncoderLayer(config.embed_size, config.num_heads, dim_feedforward = config.hidden_size, dropout = config.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, config.num_layers)
        self.encoder = nn.Embedding(config.ntokens, config.embed_size, padding_idx = config.padding_idx)
        self.ninp = config.embed_size
        self.dense = nn.Linear(config.embed_size, config.embed_size)
        self.decoder = nn.Linear(config.embed_size, 1)
        self.pool = nn.AvgPool1d(kernel_size = 128, stride = 128)

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
        output = F.relu(self.dense(output))
        output = self.decoder(output)
        return output
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=2048):
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
    config = wandb.config

    chunk_size = config.max_btch

    total_tokens = epoch * len(train_dataloader) * train_dataloader.batch_size * 128 * 1e-6
    total_compute = 6 * wandb.config.params * epoch * len(train_dataloader) * train_dataloader.batch_size * 1e-6

    start_time = time.time()
    
    scaler = GradScaler(init_scale=8192) 

    for i, batch in enumerate(train_dataloader):
        
        optimizer.zero_grad()

        data_batch, target_batch = batch[0], batch[1]
        data_chunks = torch.split(data_batch,chunk_size)
        target_chunks = torch.split(target_batch,chunk_size)
        log_loss = 0.

        for j, data in enumerate(data_chunks):
            
            target = target_chunks[j]

            data = data.to(wandb.config.device)
            target = target.to(wandb.config.device)

            src_padding_mask = (data != wandb.config.padding_idx).transpose(0, 1)
            
            with autocast(False):
                output = model(data, src_key_padding_mask = src_padding_mask) 
                loss = criterion(output, target)
                loss = loss / len(data_chunks)
            
            scaler.scale(loss).backward()
            
            log_loss += loss.item()
        
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100000)

        scaler.step(optimizer)
        scaler.update()

        total_loss += log_loss
        log_interval = 10
        total_tokens += train_dataloader.batch_size * 128 * 1e-6
        total_compute += 6 * wandb.config.params * train_dataloader.batch_size * 1e-6
      
        wandb.log({"train_loss": log_loss})
        wandb.log({"grad_norm": grad_norm})
        wandb.log({"lr": optimizer.param_groups[0]['lr']})
        wandb.log({"epoch": epoch})
        wandb.log({"n_tokens": total_tokens})
        wandb.log({"compute": total_compute})
      
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            wandb.log({"batch_time":  elapsed / log_interval})
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
    chunk_size = config.max_btch
    total_output = Tensor([]).to(config.device)
    total_target = Tensor([]).to(config.device)

    with torch.no_grad():

            for i, batch in enumerate(val_dataloader):
                
                data_batch, target_batch = batch[0], batch[1]
                data_chunks = torch.split(data_batch,chunk_size)
                target_chunks = torch.split(target_batch,chunk_size)
                    
                for j, data in enumerate(data_chunks):
            
                    target = target_chunks[j]

                    data = data.to(config.device)
                    target = target.to(config.device)

                    data = data.type(torch.IntTensor).to(config.device)
                    target = target.type(torch.FloatTensor).to(config.device)
                    target = target.view((target.shape[0],1,1))
                    src_key_padding_mask = data.eq(35)
                    src_key_padding_mask = src_key_padding_mask.permute(1,0)
                    output = eval_model(data, src_key_padding_mask = src_key_padding_mask)
                    total_loss += criterion(output, target).item()/len(data_chunks)
                    
                    total_output = torch.cat((total_output, output), dim=0)
                    total_target = torch.cat((total_target, target), dim=0)

    total_output = total_output.cpu().numpy()
    total_target = total_target.cpu().numpy()
    
    return total_loss / (len(val_dataloader)), total_output, total_target

