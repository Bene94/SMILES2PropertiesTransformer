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

@click.option('--model_name', default='220512-142153', help='Name of the model')
@click.option('--data_path', default='brouwer', help='Path to the data')

@click.option('--batch_size', default=32, help='Batch size')
@click.option('--epochs', default=50, help='Number of epochs')
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--weight_decay', default=0.0, help='Weight decay')

@click.option('--cuda', default=1, help='Use cuda')
@click.option('--local', default=True, help='Use local')

@click.option('--one_out', default=False, help='Use leave one out validation [DEPRECATED]')
@click.option('--wandb_project', default='FT_Paper', help='WandB user name')



def main(model_name, data_path, batch_size, epochs, lr, weight_decay, cuda, local, one_out, wandb_project):

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

    if one_out:
        if local:
            data_path = os.path.join('../data/' + config.data_path + '/')
        else:
            data_path = os.path.join('/mnt/xprun/data/' + config.data_path + '/')

        comp_dataset = gamma_dataset(data_path, '', config)

        outer_loop = len(comp_dataset)
        val_loss_array = np.zeros((outer_loop,config.epoch))
        val_prediction_array = np.zeros((outer_loop,config.epoch))
        val_target_array = np.zeros((outer_loop,config.epoch))
    
    else:
        training_data = load_data_full(config,local,test=False)
        val_dataloader_list = []
        outer_loop = 1

        
    

    for i in range(0,outer_loop):
        
        if one_out:
            # load data new to resett suffle
            val_dataset = deepcopy(comp_dataset)
            train_dataset = deepcopy(comp_dataset)

            diff = True

            if diff:
                comp = torch.Tensor([comp_dataset.data.solute_idx[i], comp_dataset.data.solvent_idx[i]])
                # find all other lines with the same comp in comp_dataset.train_data
                comp_index = [i for i in range(len(comp_dataset.data)) if torch.all(torch.Tensor([comp_dataset.data.solute_idx[i], comp_dataset.data.solvent_idx[i]]) == comp)]
                
                # remove the comp from the train_data
                train_dataset.data.drop(comp_index, inplace=True)
                train_dataset.data.reset_index(inplace=True)
                train_dataset.train_target = np.delete(train_dataset.train_target,comp_index,axis=0)
                train_dataset.xT = np.delete(train_dataset.xT,comp_index,axis=0)
                train_dataset.smile_index = np.delete(train_dataset.smile_index,comp_index,axis=0)
                train_dataset.index = np.delete(train_dataset.index,comp_index,axis=0)

                # select the comp from the val_data
                val_dataset.data = comp_dataset.data.iloc[comp_index]
                val_dataset.data.reset_index(inplace=True)
                val_dataset.train_target = comp_dataset.train_target[comp_index]
                val_dataset.xT = comp_dataset.xT[comp_index]
                val_dataset.smile_index = comp_dataset.smile_index[comp_index]
                val_dataset.index = comp_dataset.index[comp_index]

            else:
                val_dataset.train_data = comp_dataset.train_data[i,:].unsqueeze(0)
                val_dataset.train_target = comp_dataset.train_target[i,:].unsqueeze(0)
                train_dataset.train_data = torch.cat((comp_dataset.train_data[:i,:], comp_dataset.train_data[i+1:,:]))
                train_dataset.train_target = torch.cat((comp_dataset.train_target[:i,:],comp_dataset.train_target[i+1:,:]))

            training_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
            val_data = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
            
            val_dataloader_list = [val_data]

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

            if not one_out:

                val_loss, val_out, val_target, __ = evaluate(model, training_data, criterion, config)
                wandb.log({"val_0_loss": val_loss})
                val_loss, val_out, val_target, __  = evaluate(model, training_data, criterion, config)
                wandb.log({"val_1_loss": val_loss})

                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    .format(epoch, (time.time() - epoch_start_time),
                                                val_loss))
                print('-' * 89)
            else:
                
                val_loss, val_out, val_target, __ = evaluate(model, val_data, criterion, config)
                val_loss_array[i,epoch] = np.mean(val_loss)
                val_prediction_array[i,epoch] = np.mean(val_out)
                val_target_array[i,epoch] = np.mean(val_target)

                wandb.log({"I": i})
                wandb.log({"val_loss": val_loss_array[i,epoch]})
        wandb.log({"val_loss_ft": val_loss})

    if one_out:
        wandb.log({"val_loss_log":np.mean(val_loss_array,axis=0)}) 

                ## End Training

    # save val arrays
    if one_out:
        np.save(path_temp + 'val_loss_array_' + name + '.npy', val_loss_array)
        np.save(path_temp + 'val_prediction_array_' + name + '.npy', val_prediction_array)
        np.save(path_temp + 'val_target_array_' + name + '.npy', val_target_array)
    
    torch.save(model.state_dict(), path_model + config.xp_name +'_fine.pth')
    # save config dict with pickle
    with open(path_model + config.xp_name + '_fine.pkl', 'wb') as f:
        pickle.dump(config, f)





if __name__ == '__main__':
    main()
