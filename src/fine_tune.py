import datetime
from email.policy import default
import pickle
import os
from copy import deepcopy

import torch
from torch._C import _log_api_usage_once
import torch.nn as nn
import wandb
import click

from transprop.nn_dataloader import *
from transprop.trainer import *
import transprop.minGPT as minGPT
from transprop.cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from transprop.load_model import *
from transprop.config import *

from plot.plot_results import *

@click.command()

@click.option('--model_name', default='model_512_cosmo', help='Name of the model')
@click.option('--data_path', default='brouwer', help='Path to the data')

@click.option('--batch_size', default=32, help='Batch size')
@click.option('--epochs', default=1, help='Number of epochs')
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--weight_decay', default=0.0, help='Weight decay')


@click.option('--cuda', default=1, help='Use cuda')
@click.option('--local', default=True, help='Use local')

@click.option('--wandb_project', default='FT_Paper', help='WandB user name')
@click.option('--no_val', default=False, help='Do not use validation set')


def main(model_name, data_path, batch_size, epochs, lr, weight_decay, cuda, local, wandb_project, no_val):

    name = model_name

    if os.environ.get('XPRUN_NAME') is not None:
        local = False
        path_temp = "/mnt/xprun/temp/"
        path_model = "/mnt/xprun/out/"
        path_wandb = "/mnt/xprun/wandb/"
        xp_name = os.environ['XPRUN_NAME']
    else:
        local = True
        path_temp = '../temp/'
        path_model = '../Models/'
        path_wandb = '../wandb/'
        xp_name = "local_test"

    if cuda == 1:
        device = 'cuda'
    else:
        device = 'cpu'
    
    model, config = load_model(path_model,model_name)
    model = model.to(config.device)

    # overide old config
    config.data_path = data_path
    config.batch_size = batch_size
    config.epoch = epochs
    config.lr = lr
    config.wdcay = weight_decay
    config.device = device
    config.local = local
    config.xp_name = xp_name

    wandb.init(project=wandb_project, name=name, config=config)
    wandb.watch(model)

    ## set up scheduler
    criterion = nn.MSELoss()

    optimizer = model.configure_optimizers(config)

    total_steps = 100000 * config.epoch
        
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_steps, gamma=1)

    epoch_start = 0
    best_val_loss = float('inf')


    if  no_val:
        training_data = load_data_full(config,local,test=False)
        val_dataloader_list = []
    else:
        training_data, val_0_data, val_1_data, val_2_data = load_data(config,local,test=False)
        val_dataloader_list = []
        val_dataloader_list.append(val_0_data)
        val_dataloader_list.append(val_1_data)
        val_dataloader_list.append(val_2_data)

    model, __ = load_model(path_model,model_name)
    model = model.to(config.device)

    optimizer = model.configure_optimizers(config)

    total_steps = len(training_data) * config.epoch
        
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_steps, gamma=1)

    epoch_start = 0
    best_val_loss = float('inf')

    # init the gradScaler
    scaler = GradScaler(init_scale=8192) 

    for epoch in range(epoch_start, config.epoch):

        epoch_start_time = time.time()

        train(model, criterion, optimizer, training_data, val_dataloader_list, scheduler, epoch, wandb, scaler)
        
        torch.cuda.empty_cache()

    torch.save(model.state_dict(), path_model + config.xp_name +'_fine.pth')
    # save config dict with pickle
    with open(path_model + config.xp_name + '_fine.pkl', 'wb') as f:
        pickle.dump(config, f)

if __name__ == '__main__':
    main()
