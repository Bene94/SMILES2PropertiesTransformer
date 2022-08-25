import datetime
import pickle
import os
from numpy.random.mtrand import random
from six import with_metaclass

import torch
import torch.nn as nn
import wandb
import click

from transprop.nn_dataloader import *
from plot.plot_results import *
from transprop.trainer import *
import transprop.minGPT as minGPT
from transprop.cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from transprop.load_model import *
from transprop.config import *

@click.command()

@click.option('--emb', default=512, help='Embedding size')
@click.option('--hid_fac', default=4, help='Hidden layer size')
@click.option('--nlay', default=2, help='Number of transformer layers')
@click.option('--nhead', default=16, help='Number of heads')
@click.option('--wdecay', default=0.0, help='Weight decay')
@click.option('--drp', default=0.0, help='Dropout rate')

@click.option('--mode', default="reg", help='Determines the mode: reg: does a regresstion; class: does a classification - deprecated')
@click.option('--bins', default=2000, help='Determins the number of bins in the clasifcation mode')
@click.option('--aug', default=0, help='Determines if the data is augmented')

@click.option('--lr', default= 0.001, help='Learning rate')
@click.option('--epo', default=50, help='Number of epochs')
@click.option('--btch', default=1024, help='Batchsize')
@click.option('--max_btch', default=512, help='Maximum batch size for the GPU, will use gradient accumulation')

@click.option('--warmup_epo', default=5, help='Number of warmup epochs')
@click.option('--warmup_lr', default=100, help='Reduciton of LR in the warmup')
@click.option('--warmup_cycle', default=1, help='Number of warmup cycels')
@click.option('--warmup_gamma', default=1.0, help='Warmup gamma')
@click.option('--stop_epo', default=0, help='Number of epochs to stop warmup')

@click.option('--data', default='inf_t_cosmo', help='Location of dataset')

@click.option('--cuda', default=True, help='Using GPU')
@click.option('--log_name', default='test', help='name in the log')
@click.option('--test', default=0, help='If true smale dataset is used')
@click.option('--project', default='test', help='wandb project name')

@click.option('--shift', default=0, help='Shift the data')
@click.option('--xt', default=1, help='If xT should be used')
@click.option('--noval', default=0, help='All data is used for training, no validation takes place')


def main(emb, hid_fac, nlay, nhead, drp, lr, epo, btch, data, wdecay, max_btch, cuda, log_name, warmup_epo, warmup_lr, warmup_cycle, warmup_gamma, test, mode, bins, aug, shift, xt, stop_epo, noval, project):
    
    name = str(emb) + '_' + str(nlay) + '_' + str(nhead) + '_' + '{:.0e}'.format(drp) + '_' + '{:.0e}'.format(wdecay) + '_' + '{:.0e}'.format(lr) +  '_' + str(btch) + '_' + str(epo)
    
    print('-' * 89)
    print('Beginn Script...')
    print('-' * 89)

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
        if log_name != '':
            xp_name = log_name
        else:
            xp_name = 'local_test' + str(random())

    if test == 1:
        test = True
    else:
        test = False

    test = False

    if cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    criterion = nn.MSELoss()

    if stop_epo == 0:
        stop_epo = epo

    vocab_size =  100

    config = NN_config(xp_name=xp_name, device=device, criterion=criterion, padding_idx=0, 
        vocab_size=vocab_size, block_size=128, embed_size=emb, hidden_factor=hid_fac, num_layers=nlay, 
        num_heads=nhead, dropout=drp, lr=lr, warmup_lr = warmup_lr, warmup_cycle=warmup_cycle, betas=[0.90 , 0.95],
        weight_decay=wdecay, data_path=data, path_temp=path_temp, path_model=path_model, batch_size=btch, max_btch=max_btch, epoch=epo, warmup_epochs=warmup_epo, 
        mode=mode, bins=bins, bound=20, shift=shift, test=test, xT=xt)

    ## load training and validation data
    print('-' * 89)
    print('Loading Data...')
    print('-' * 89)
  
    if noval==1:
        training_data = load_data_full(config,local,test=test)
        val_data_list = []
    else:
        training_data, val_0_data, val_1_data, val_2_data = load_data(config,local,test=test)
        val_data_list = []
        val_data_list.append(val_0_data) 
        val_data_list.append(val_1_data)
        val_data_list.append(val_2_data)

    # init the gradScaler
    scaler = GradScaler(init_scale=8192) 

    # see if file with name xp_name exists
    if os.path.isfile(path_temp + config.xp_name + '_epoch.pkl'):
        model, config, optimizer, scheduler, epoch_start = load_checkpoint(config)

    else:
        model = minGPT.GPT(config)
        model = model.to(config.device)
        config.params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        optimizer = model.configure_optimizers(config)

        ## set up scheduler
        total_steps = len(training_data) * (config.epoch +1)
        min_lr = config.lr / config.warmup_lr
        warumup_steps = int(total_steps * (config.warmup_epochs / config.epoch))
        first_cycle_steps = int(total_steps / config.warmup_cycle)
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=first_cycle_steps, cycle_mult=1.0, max_lr=config.lr, min_lr=min_lr, warmup_steps=warumup_steps, gamma=warmup_gamma)

        epoch_start = 0

        config.id = wandb.util.generate_id()

    if test:
        project = 'Test'
    else:
        project = project

    wandb.init(project=project, name=name, config=config, resume="allow", id=config.id)
    wandb.watch(model)
    wandb.log({"xp_name": config.xp_name})
    wandb.log({"mode": mode})

    print('-' * 89)
    print('Training...')
    print('-' * 89)

    ## train model
    best_val_loss = float("inf")
    best_model = None

    overall_start_time = time.time()
    
    for epoch in range(epoch_start, stop_epo + 1):

        epoch_start_time = time.time()

        train(model, criterion, optimizer, training_data, val_data_list, scheduler, epoch, wandb, scaler)

        torch.cuda.empty_cache()

        # save checkpoint to resume training
        save_checkpoint(model, config, epoch, optimizer, scheduler)

    if not noval:
        val_loss, val_out, val_target, __ = evaluate(model, val_0_data, criterion, config)
        wandb.log({"val_0_loss": val_loss})
        val_loss, val_out, val_target, __ = evaluate(model, val_1_data, criterion, config)
        wandb.log({"val_1_loss": val_loss})
        val_loss, val_out, val_target, __ = evaluate(model, val_2_data, criterion, config)
        wandb.log({"val_2_loss": val_loss})

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            .format(epoch, (time.time() - epoch_start_time),
                                        val_loss))
        print('-' * 89)
        
    best_model = model  

    ## End Training

    print('-' * 89)
    print('| End of training | time: {:5.2f}s |'.format((time.time() - overall_start_time)))
    print('-' * 89)

    torch.save(best_model.state_dict(), path_model + config.xp_name +'.pth')
    # save config dict with pickle
    with open(path_model + config.xp_name + '.pkl', 'wb') as f:
        pickle.dump(config, f)

    delete_checkpoint(config)

    
def save_checkpoint(model, config, epoch, optimizer, scheduler):
    """
    Saves model checkpoint.
    """
    path = config.path_temp

    torch.save(model.state_dict(), path + config.xp_name + '.pth')
    # save config dict with pickle
    with open(path + config.xp_name + '.pkl', 'wb') as f:
        pickle.dump(config, f)
    # save optimizer state dict with pickle
    with open(path + config.xp_name + '_optimizer.pkl', 'wb') as f:
        pickle.dump(optimizer.state_dict(), f)
    # save scheduler state dict with pickle
    with open(path + config.xp_name + '_scheduler.pkl', 'wb') as f:
        scheduler_dict = scheduler.dump_state_dict()
        pickle.dump(scheduler_dict, f)
    with open (path + config.xp_name + '_epoch.pkl', 'wb') as f:
        pickle.dump(epoch, f)


def load_checkpoint(config):
    """
    Loads model checkpoint.
    """
    path = config.path_temp

    model = minGPT.GPT(config)
    model.load_state_dict(torch.load(path + config.xp_name + '.pth'))
    model.to(config.device)
    with open(path + config.xp_name + '.pkl', 'rb') as f:
        config = pickle.load(f)
    with open(path + config.xp_name + '_optimizer.pkl', 'rb') as f:
        optimizer = model.configure_optimizers(config)
        optimizer.load_state_dict(pickle.load(f))
    with open(path + config.xp_name + '_scheduler.pkl', 'rb') as f:
        scheduler_dict = pickle.load(f)
        scheduler = CosineAnnealingWarmupRestarts(optimizer)
        scheduler.load_state_dict(scheduler_dict)
    with open(path + config.xp_name + '_epoch.pkl', 'rb') as f:
        epoch = pickle.load(f)
    return model, config, optimizer, scheduler, epoch

def delete_checkpoint(config):
    """
    Deletes model checkpoint.
    """
    path = config.path_temp

    os.remove(path + config.xp_name + '.pth')
    os.remove(path + config.xp_name + '.pkl')
    os.remove(path + config.xp_name + '_optimizer.pkl')
    os.remove(path + config.xp_name + '_scheduler.pkl')
    os.remove(path + config.xp_name + '_epoch.pkl')


def plotting(val_target, val_out, local, log_name, epoch, val_loss, train_loss, train_out, train_target):


    if local:
        path = '/home/bene/NNGamma/Plot/'
    else:
        path = "/mnt/xprun/plot/"

    name_plot = log_name + '_' +  str(epoch) + '_val_' + '{:.1e}'.format(val_loss) +'.png'

    val_target = val_target.squeeze()
    val_out = val_out.squeeze()

    make_histogram(val_out, val_target, name_plot, path)
    make_heatmap(val_out, val_target, name_plot, path)
    
    train_target = train_target.squeeze()
    train_out = train_out.squeeze()
    
    name_plot = log_name + '_' + str(epoch) + '_train_' + '{:.1e}'.format(train_loss) + '.png'
    
    make_histogram(train_out, train_target, name_plot, path)
    make_heatmap(train_out, train_target, name_plot, path)


if __name__ == '__main__': 
    main()
