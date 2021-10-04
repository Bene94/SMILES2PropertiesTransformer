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

@click.option('--model_name', default='211003-111953', help='Name of the model')
@click.option('--data_path', default='data_exp', help='Path to the data')

@click.option('--batch_size', default=32, help='Batch size')
@click.option('--epochs', default=1, help='Number of epochs')
@click.option('--lr', default=1e-5, help='Learning rate')
@click.option('--weight_decay', default=0.0, help='Weight decay')

@click.option('--cuda', default=True, help='Use cuda')
@click.option('--local', default=False, help='Use local')


def main(model_name, data_path, batch_size, epochs, lr, weight_decay, cuda, local):

    name = model_name

    # load model and config
    if local:
        path_temp = '../temp/'
        path_model = '../Models/'
        xp_name = "local_test"
    else:
        path_temp = "/mnt/xprun/temp/"
        path_model = "/mnt/xprun/out/"
        xp_name = os.environ['XPRUN_NAME']

    if cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    local = False

    path_temp = "/mnt/xprun/temp/"
    path_model = "/mnt/xprun/out/"
    xp_name = os.environ['XPRUN_NAME']


    
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

    wandb.init(project='GNN_001_FT_step', entity='bene94', name=name, config=config)
    wandb.watch(model)

    if local:
        data_path = os.path.join('/home/bene/NNGamma/' + config.data_path + '/')
    else:
        data_path = os.path.join('/mnt/xprun/' + config.data_path + '/')

    comp_dataset = gamma_dataset(data_path, '', config)

    ## determin the datasets:

    sample_sizes = np.array([10, 100, 1000])
    min_epoch = 2
    min_sampels = 10
    sampels = np.array([1000, 100, 10])
    epochs = np.array([2, 2, 2])

    # create the datasets

    train_dataloader_list =  []
    val_dataloader_list = []

    n_data = len(comp_dataset)

    
    # baseline loss of original model

    comp_dataloader = torch.utils.data.DataLoader(comp_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    model, __ = load_model(path_model,model_name)
    model = model.to(config.device)

    optimizer = model.configure_optimizers(config)

    criterion = nn.MSELoss()
    loss, __, __ =  evaluate(model, comp_dataloader, criterion, config)
    print("Reference Loss: " + str(loss))

    for i in range(len(sample_sizes)):
        train_dataloader_list.append([])
        val_dataloader_list.append([])
 
        for j in range(sampels[i]):
            val_dataset = deepcopy(comp_dataset)
            train_dataset = deepcopy(comp_dataset)

            # fix random state
            np.random.seed((i,j))
            idx = np.random.choice(n_data, size=sample_sizes[i], replace=False)
            # get all other indices
            idx_other = np.array(range(n_data))
            idx_other = np.delete(idx_other, idx)

            train_dataset.train_data = comp_dataset.train_data[idx,:]
            train_dataset.train_target = comp_dataset.train_target[idx,:]

            val_dataset.train_data = comp_dataset.train_data[idx_other,:]
            
            train_dataloader_list[i].append(DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0))
            val_dataloader_list[i].append(DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0))

    
    val_loss_array = []
    val_prediction_array = []
    val_target_array = []

    ## run the models
    for i in range(len(sample_sizes)):
        val_loss_array.append([])
        val_prediction_array.append([])
        val_target_array.append([])

        for j in range(sampels[i]):
            
            print("i: " + str(i)+ " j: " + str(j))
            val_loss_array[i].append([])
            val_prediction_array[i].append([])
            val_target_array[i].append([])

            model, __ = load_model(path_model,model_name)
            model = model.to(config.device)

            optimizer = model.configure_optimizers(config)

            criterion = nn.MSELoss()

            training_data = train_dataloader_list[i][j]
            validation_data = val_dataloader_list[i][j]

            total_steps = len(training_data) * epochs[i]
                
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_steps, gamma=1)
            
            best_val_loss = float('inf')



            for epoch in range(0, epochs[i]):

                val_loss_array[i][j].append([])
                val_prediction_array[i][j].append([])
                val_target_array[i][j].append([])

                epoch_start_time = time.time()

                train(model, criterion, optimizer, training_data, scheduler, epoch, wandb)
                
                torch.cuda.empty_cache()

                val_loss_array[i][j][epoch], val_prediction_array[i][j][epoch], val_target_array[i][j][epoch] = evaluate(model, validation_data, criterion, config)
                
                wandb.log({"I": i})
                wandb.log({"val_loss_log":np.mean(val_loss_array[i][j][epoch])}) 

                ## End Training

        # save val arrays

        np.save(path_temp + 'val_loss_array_' + name + '.npy', val_loss_array)
        np.save(path_temp + 'val_prediction_array_' + name + '.npy', val_prediction_array)
        np.save(path_temp + 'val_target_array_' + name + '.npy', val_target_array)






if __name__ == '__main__':
    main()
