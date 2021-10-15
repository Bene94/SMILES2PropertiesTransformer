import datetime
import pickle
import os
from copy import deepcopy

import torch
from torch._C import _log_api_usage_once
import torch.nn as nn
import wandb
import click


from nn_model import * 
from nn_dataloader import *
from plot_results import *
from trainer import *
import minGPT
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from load_model import *
from config import *

@click.command()

@click.option('--model_name', default='211004-141758', help='Name of the model')
@click.option('--data_path', default='data_exp', help='Path to the data')

@click.option('--batch_size', default=32, help='Batch size')
@click.option('--epochs', default=5, help='Number of epochs')
@click.option('--lr', default=1e-5, help='Learning rate')
@click.option('--weight_decay', default=0.0, help='Weight decay')

@click.option('--cuda', default=True, help='Use cuda')

@click.option('--mult', default=100, help='Uses multibel val/train splits')


def main(model_name, data_path, batch_size, epochs, lr, weight_decay, cuda, mult):

    name = model_name

    if os.environ.get('XPRUN_NAME') is not None:
        local = False
        path_temp = "/mnt/xprun/temp/"
        path_model = "/mnt/xprun/out/"
        path_wandb = "/mnt/xprun/wandb/"
        xp_name = os.environ['XPRUN_NAME']
    else:
        local = True
        path_temp = '../out_fine_tuen/'
        path_model = '../Models/'
        path_wandb = '../wandb/'
        xp_name = "local_test"

    if cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    
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

    wandb.init(project='GNN_001_FT', entity='bene94', name=name, config=config)
    wandb.watch(model)



    
    outer_loop = mult
    
    val_predction_0 = np.array([])
    val_predction_1 = np.array([])
    val_predction_2 = np.array([])

    val_target_0 = np.array([])
    val_target_1 = np.array([])
    val_target_2 = np.array([])


    for i in range(0,outer_loop):

        model, __ = load_model(path_model,model_name)
        model = model.to(config.device)
        wandb.watch(model)

        ## set up scheduler
        criterion = nn.MSELoss()

        optimizer = model.configure_optimizers(config)

        total_steps = 100000 * config.epoch
            
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_steps, gamma=1)

        epoch_start = 0
        
        config.data_path = data_path + '/'  + str(i)
        training_data, val_0_data, val_1_data, val_2_data = load_data(config,local,test=False)

        for epoch in range(epoch_start, config.epoch):

            epoch_start_time = time.time()

            train(model, criterion, optimizer, training_data, scheduler, epoch, wandb)
            
            torch.cuda.empty_cache()

            # evaluate the 3 validation sets

        
        temp_val_loss, temp_val_prediction, temp_val_target = evaluate(model, val_0_data, criterion, config)

        val_predction_0 = np.concatenate((val_predction_0, temp_val_prediction), axis=0)
        val_target_0 = np.concatenate((val_target_0, temp_val_target), axis=0)

        temp_val_loss, temp_val_prediction, temp_val_target = evaluate(model, val_1_data, criterion, config)

        val_predction_1 = np.concatenate((val_predction_1, temp_val_prediction), axis=0)
        val_target_1 = np.concatenate((val_target_1, temp_val_target), axis=0)

        temp_val_loss, temp_val_prediction, temp_val_target = evaluate(model, val_2_data, criterion, config)

        val_predction_2 = np.concatenate((val_predction_2, temp_val_prediction), axis=0)
        val_target_2 = np.concatenate((val_target_2, temp_val_target), axis=0)
    
        wandb.log({"I": i})
    

    np.save(path_temp + 'val_prediction_array_0_' + name + '.npy', val_predction_0)
    np.save(path_temp + 'val_target_array_0' + name + '.npy', val_target_0)

    np.save(path_temp + 'val_prediction_array_1' + name + '.npy', val_predction_1)
    np.save(path_temp + 'val_target_array_1' + name + '.npy', val_target_1)

    np.save(path_temp + 'val_prediction_array_2' + name + '.npy', val_predction_2)
    np.save(path_temp + 'val_target_array_2' + name + '.npy', val_target_2)
        

if __name__ == '__main__':
    main()
