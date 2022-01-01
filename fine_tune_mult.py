import datetime
import pickle
import os
from copy import deepcopy

import torch
from torch._C import _log_api_usage_once
import torch.nn as nn
from torch.optim import lr_scheduler
import wandb
import click

from transprop.nn_dataloader import *
from transprop.trainer import *
import transprop.minGPT as minGPT
from transprop.cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from transprop.load_model import *
from transprop.config import *

@click.command()

@click.option('--model_name', '-m', default='211220-192228', help='Name of the model')
@click.option('--data_path', '-p',default='data_exp', help='Path to the data')
@click.option('--exp_name', '-n',default='', help='Name of the experiment')

@click.option('--batch_size', '-b', default=32, help='Batch size')
@click.option('--epochs', '-e',default=20, help='Number of epochs')
@click.option('--lr', '-l',default=1e-5, help='Learning rate')
@click.option('--weight_decay', default=0.0, help='Weight decay')

@click.option('--cuda', default=True, help='Use cuda')

@click.option('--mult', '-x',default=2, help='Uses multibel val/train splits')
@click.option('--ow', '-ow',default=0, help='if 1, overwrites existing outputs')
@click.option('--lval', '-lval',default=0, help='if 1, log validation loss every epoch') 



def main(model_name, data_path, exp_name, batch_size, epochs, lr, weight_decay, cuda, mult, ow, lval):

    name = model_name

    if os.environ.get('XPRUN_NAME') is not None:
        
        print("Run on XPRUN")

        local = False

        path_temp = "/mnt/xprun/temp/"
        path_model = "/mnt/xprun/out/"
        path_wandb = "/mnt/xprun/wandb/"
        xp_name = os.environ['XPRUN_NAME']
    else:
        print("Run on local machine")
        local = True
        path_temp = '../out_fine_tuen/'
        path_model = '../Models/'
        path_wandb = '../wandb/'
        xp_name = "local_test"

    if cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if exp_name != '':
        xp_name = exp_name
    
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

    lr_schedule = 'cosine'

    wandb.init(project='GNN_001_FT_mult', entity='bene94', name=name, config=config)
    wandb.watch(model)

    print("=" * 50)
    print("Start of training")
    print("=" * 50)
    
    outer_loop = mult
    
    # check if checkpoints exist
    if os.path.exists(path_temp + xp_name + '/i.npy') and not ow == 0:
        print("Loading checkpoint")
        i_start = np.load(path_temp + xp_name + '/i.npy')
    else:
        i_start = 0

    # allocate memory for epo val loss

    epo_val_loss = np.zeros((outer_loop,epochs,3))

    for i in range(i_start,outer_loop):

        wandb.log({'outer_loop': i})
        print("outer_loop: ", i)

        model, __ = load_model(path_model,model_name)
        model = model.to(config.device)
        #wandb.watch(model)


        ## set up scheduler
        criterion = nn.MSELoss()

        optimizer = model.configure_optimizers(config)
        epoch_start = 0
        
        config.data_path = data_path + '/'  + str(i)

        training_data, val_0_data, val_1_data, val_2_data = load_data(config,local,test=False)

        # load validaton data with larger batch size
        temp_btch_size = config.batch_size
        config.batch_size = 256
        __ , val_0_data, val_1_data, val_2_data = load_data(config,local,test=False)
        config.batch_size = temp_btch_size

        val_data_list = []
        val_data_list.append(val_0_data) 
        val_data_list.append(val_1_data)
        val_data_list.append(val_2_data)

        if lr_schedule == 'cosine':
                    ## set up scheduler
            total_steps = len(training_data) * (config.epoch +1)
            min_lr = config.lr / 100
            warumup_steps = int(total_steps * (5 / config.epoch))
            first_cycle_steps = int(total_steps / 1)
            scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=first_cycle_steps, cycle_mult=1.0, max_lr=config.lr, min_lr=min_lr, warmup_steps=warumup_steps, gamma=1)

        else:
            total_steps = 100000 * config.epoch
              
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_steps, gamma=1)



        for epoch in range(epoch_start, config.epoch):

            epoch_start_time = time.time()

            train(model, criterion, optimizer, training_data, [], scheduler, epoch, wandb)
            
            torch.cuda.empty_cache()

            if lval == 1:
                epo_val_loss_0, __ , __ , __ = evaluate(model, val_0_data, criterion, config) 
                epo_val_loss_1, __ , __ , __ = evaluate(model, val_1_data, criterion, config)
                epo_val_loss_2, __ , __ , __ = evaluate(model, val_2_data, criterion, config)

                epo_val_loss[i,epoch,0] = epo_val_loss_0
                epo_val_loss[i,epoch,1] = epo_val_loss_1
                epo_val_loss[i,epoch,2] = epo_val_loss_2

                wandb.log({'epoch_val_loss_0': epo_val_loss_0, 'epoch_val_loss_1': epo_val_loss_1, 'epoch_val_loss_2': epo_val_loss_2})
                wandb.log({'epoch': epoch})

            # evaluate the 3 validation sets

        
        temp_val_loss, val_predction_0, val_target_0, val_input_0 = evaluate(model, val_0_data, criterion, config)
        
        wandb.log({"val_0_ft": temp_val_loss})

        temp_val_loss, val_predction_1, val_target_1, val_input_1  = evaluate(model, val_1_data, criterion, config)

        wandb.log({"val_1_ft": temp_val_loss})

        temp_val_loss, val_predction_2, val_target_2, val_input_2  = evaluate(model, val_2_data, criterion, config)
    
        wandb.log({"val_2_ft": temp_val_loss})
    
        if not os.path.exists(path_temp + xp_name):
            os.makedirs(path_temp + xp_name)

        np.save(path_temp + xp_name + '/i.npy', i)
        
        np.save(path_temp + xp_name + '/val_predction_0_'+ str(i) +'.npy', val_predction_0)
        np.save(path_temp + xp_name + '/val_predction_1_'+ str(i) +'.npy', val_predction_1)
        np.save(path_temp + xp_name + '/val_predction_2_'+ str(i) +'.npy', val_predction_2)

        np.save(path_temp + xp_name + '/val_input_0_'+ str(i) +'.npy', val_input_0[2])
        np.save(path_temp + xp_name + '/val_input_1_'+ str(i) +'.npy', val_input_1[2])
        np.save(path_temp + xp_name + '/val_input_2_'+ str(i) +'.npy', val_input_2[2])

        np.save(path_temp + xp_name + '/val_target_0_'+ str(i) +'.npy', val_target_0)
        np.save(path_temp + xp_name + '/val_target_1_'+ str(i) +'.npy', val_target_1)
        np.save(path_temp + xp_name + '/val_target_2_'+ str(i) +'.npy', val_target_2)

    np.save(path_temp + xp_name + '/epo_val_loss', epo_val_loss)
        
if __name__ == '__main__':
    main()
