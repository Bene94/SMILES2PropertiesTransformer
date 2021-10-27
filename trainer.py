import math
import time
import numpy as np

import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.modules import activation




def train(model, criterion, optimizer, train_dataloader, val_dataloader_list, scheduler, epoch, wandb):
    
    model.train() # Turn on the train mode
    total_loss = 0.
    config = wandb.config

    chunk_size = config.max_btch

    total_tokens = epoch * len(train_dataloader) * train_dataloader.batch_size * 128 * 1e-6
    total_compute = 6 * wandb.config.params * epoch * len(train_dataloader) * train_dataloader.batch_size * 1e-6

    start_time = time.time()
    
    scaler = GradScaler(init_scale=8192) 

    n_steps = len(train_dataloader)

    iter_val_dataloader_list  = []
    val_step = []  
    for i in range(len(val_dataloader_list)):
        iter_val_dataloader_list.append(iter(val_dataloader_list[i]))
        val_step.append(int(np.ceil(len(train_dataloader) /len(val_dataloader_list[i]))))


    for i, batch in enumerate(train_dataloader):
        
        optimizer.zero_grad()

        target_batch, data_batch = batch[0], batch[1]
        smile = data_batch[0]
        xt = data_batch[1]
        
        target_chunks = torch.split(target_batch,chunk_size)
        smile_chunks = torch.split(smile,chunk_size)
        xt_chunks = torch.split(xt,chunk_size)

        log_loss = 0.

        for j in range(len(target_chunks)):
            
            target = target_chunks[j]
            smile = smile_chunks[j]
            xt = xt_chunks[j]

            xt[:,0] = xt[:,0] - 0.5
            xt[:,1] = xt[:,1] / 298.5 -1.


            smile = smile.type(torch.IntTensor)

            target = target.to(wandb.config.device)
            smile = smile.to(wandb.config.device)
            xt = xt.to(wandb.config.device)

            src_padding_mask = (smile != wandb.config.padding_idx).transpose(0, 1)

            with autocast(True):
                #xt = xt.type(torch.half)
                output = model(smile, xt) 
                loss = criterion(output.squeeze(), target.squeeze())
                loss = loss / len(target_chunks)
            
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

        # check if we should validate run validation
        for h , step in enumerate(val_step):
            if i % step == 0 and i > 0:
                with torch.no_grad():
                    val_loss = 0.

                    batch = next(iter_val_dataloader_list[h])
                    target_batch, data_batch = batch[0], batch[1]
                    smile = data_batch[0]
                    xt = data_batch[1]
                    
                    target_chunks = torch.split(target_batch,chunk_size)
                    smile_chunks = torch.split(smile,chunk_size)
                    xt_chunks = torch.split(xt,chunk_size)
                        
                    for j in range(len(target_chunks)):
                
                        target = target_chunks[j]
                        smile = smile_chunks[j]
                        xt = xt_chunks[j]

                        xt[:,0] = xt[:,0] - 0.5
                        xt[:,1] = xt[:,1] / 298.5 -1.

                        smile = smile.type(torch.IntTensor)

                        target = target.to(wandb.config.device)
                        smile = smile.to(wandb.config.device)
                        xt = xt.to(wandb.config.device)
                        
                        output = model(smile, xt)

                        val_loss += criterion(output.squeeze(), target.squeeze()).item()/len(target_chunks)
                    
                    val_log_name = 'val_' + str(h) + '_loss'
                    wandb.log({val_log_name : val_loss})


            


def evaluate(eval_model, val_dataloader, criterion, config):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    chunk_size = config.max_btch
    total_output = np.array([])
    total_target = np.array([])

    with torch.no_grad():

            # incude progress bar

            for i, batch in enumerate(val_dataloader):

                target_batch, data_batch = batch[0], batch[1]
                smile = data_batch[0]
                xt = data_batch[1]
                
                target_chunks = torch.split(target_batch,chunk_size)
                smile_chunks = torch.split(smile,chunk_size)
                xt_chunks = torch.split(xt,chunk_size)
                    
                for j in range(len(target_chunks)):
            
                    target = target_chunks[j]
                    smile = smile_chunks[j]
                    xt = xt_chunks[j]

                    xt[:,0] = xt[:,0] - 0.5
                    xt[:,1] = xt[:,1] / 298.5 -1.

                    smile = smile.type(torch.IntTensor)

                    target = target.to(config.device)
                    smile = smile.to(config.device)
                    xt = xt.to(config.device)
                    
                    if config.mode == 'reg':
                        target = target.view((target.shape[0],1,1))
                    else:
                        target = target.type(torch.LongTensor).to(config.device)

                    output = eval_model(smile, xt)
                    total_loss += criterion(output.squeeze(), target.squeeze()).item()/len(target_chunks)
                    
                    total_output = np.append(total_output, output.cpu().numpy())
                    total_target = np.append(total_target, target.cpu().numpy())
    
    return total_loss / (len(val_dataloader)), total_output, total_target

