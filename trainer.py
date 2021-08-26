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



def train(model, criterion, optimizer, train_dataloader, scheduler, epoch, wandb):
    
    model.train() # Turn on the train mode
    total_loss = 0.
    config = wandb.config

    chunk_size = config.max_btch

    total_tokens = epoch * len(train_dataloader) * train_dataloader.batch_size * 128 * 1e-6
    total_compute = 6 * wandb.config.params * epoch * len(train_dataloader) * train_dataloader.batch_size * 1e-6

    start_time = time.time()
    
    scaler = GradScaler(init_scale=8192) 

    n_steps = len(train_dataloader)

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
            
            with autocast(True):
                output = model(data) 
                loss = criterion(output.squeeze(), target.squeeze())
                loss = loss / len(data_chunks)
            
            scaler.scale(loss).backward()
            
            log_loss += loss.item()
        
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100000)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        ##Loss logging and display

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
                    epoch, i, len(train_dataloader), optimizer.param_groups[0]['lr'],
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

            # incude progress bar

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
                    output = eval_model(data)
                    total_loss += criterion(output.squeeze(), target.squeeze()).item()/len(data_chunks)
                    
                    total_output = torch.cat((total_output, output), dim=0)
                    total_target = torch.cat((total_target, target), dim=0)

    total_output = total_output.cpu().numpy()
    total_target = total_target.cpu().numpy()
    
    return total_loss / (len(val_dataloader)), total_output, total_target

