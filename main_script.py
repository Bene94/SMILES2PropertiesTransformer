import datetime
import pickle
import os

import torch
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

@click.option('--emb', default=512, help='Embedding size')
@click.option('--hid_fac', default=4, help='Hidden layer size')
@click.option('--nlay', default=2, help='Number of transformer layers')
@click.option('--nhead', default=8, help='Number of heads')
@click.option('--wdecay', default=0.1, help='Weight decay')
@click.option('--drp', default=0.1, help='Dropout rate')

@click.option('--mode', default="reg", help='Determines the mode: reg: does a regresstion; class: does a classification')
@click.option('--bins', default=2000, help='Determins the number of bins in the clasifcation mode')

@click.option('--lr', default= 0.0001, help='Learning rate')
@click.option('--epo', default=10, help='Number of epochs')
@click.option('--btch', default=1024, help='Batchsize')
@click.option('--max_btch', default=128, help='Maximum batch size')

@click.option('--warmup_epo', default=1, help='Number of warmup epochs')
@click.option('--warmup_lr', default=10, help='Reduciton of LR in the warmup')
@click.option('--warmup_cycle', default=1, help='Number of warmup cycels')
@click.option('--warmup_gamma', default=1.0, help='Warmup gamma')

@click.option('--set', default='data', help='Location of dataset')

@click.option('--modle_type', default="minGPT", help='Selected Modle')

@click.option('--cuda', default=True, help='Using GPU')
@click.option('--log_name', default='', help='Using GPU')
@click.option('--local' , default=False, help='Using training data from local folder')
@click.option('--test', default=False, help='If true smale dataset is used')

@click.option('--shift', default=0, help='Shift the data')

@click.option('--fine_tune', default='NO', help='load a privious modle and finetune it, name the modle to load here')


def main(emb, hid_fac, nlay, nhead, drp, lr, epo, btch, set, wdecay, local, max_btch, cuda, log_name, modle_type, warmup_epo, warmup_lr, warmup_cycle, warmup_gamma, test, mode, bins, shift, fine_tune):
    
    name = modle_type + '_' + str(emb) + '_' + str(nlay) + '_' + str(nhead) + '_' + '{:.0e}'.format(drp) + '_' + '{:.0e}'.format(wdecay) + '_' + '{:.0e}'.format(lr) +  '_' + str(btch) + '_' + str(epo)
    
    if local:
        xp_name = "NaN"
    else:
        xp_name = os.environ['XPRUN_NAME']

    if cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    criterion = nn.MSELoss()

   # check if set containts red
    if 'red' in set:
        vocab_size = 23
    else:
        vocab_size =  40

    config = NN_config(xp_name=xp_name, device=device, criterion=criterion, padding_idx=0, 
        vocab_size=vocab_size, block_size=128, embed_size=emb, hidden_factor=hid_fac, num_layers=nlay, 
        num_heads=nhead, dropout=drp, lr=lr, betas=[0.99 , 0.98], weight_decay=wdecay, data_path=set,
        batch_size=btch, max_btch=max_btch, epoch=epo, warmup_epochs=warmup_epo, 
        mode=mode, bins=bins, bound=20, shift=shift)

    ## load training and validation data

    print('-' * 89)
    print('Loading Data...')
    print('-' * 89)

    training_data, val_0_data, val_1_data = load_data(config,local,test)

    ## create model


        # load a previous model
    if not fine_tune == 'NO':
        
        if local:
            path = '/home/bene/NNGamma/Models/'
        else:
            path = "/mnt/xprun/out/"
        
        config_loaded, model = load_model(path, fine_tune)

        # set the architecure of the loaded model
        config.embed_size = config_loaded.embed_size
        config.hidden_factor = config_loaded.hidden_factor
        config.num_layers = config_loaded.num_layers
        config.num_heads = config_loaded.num_heads
        config.dropout = config_loaded.dropout
    else:
        if mode == "reg":
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        if modle_type == 'minGPT':
            model = minGPT.GPT(config)
            optimizer = model.configure_optimizers(config)
        elif modle_type == 'pytorch':
            model = TransformerModel(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    



    model = model.to(config.device)
    config.params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    wandb.init(project= 'gamma', entity='bene94', name=name, config=config)
    wandb.watch(model)

    ## set up scheduler
    total_steps = len(training_data) * config.epoch
    min_lr = config.lr / warmup_lr
    warumup_steps = int(total_steps * config.warmup_epochs / config.epoch)
    first_cycle_steps = int(total_steps / warmup_cycle)
    
    scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=first_cycle_steps, cycle_mult=1.0, max_lr=config.lr, min_lr=min_lr, warmup_steps=warumup_steps, gamma=warmup_gamma)

    ## train model
    best_val_loss = float("inf")
    best_model = None

    overall_start_time = time.time()

    for epoch in range(1, config.epoch + 1):

        epoch_start_time = time.time()

        train(model, criterion, optimizer, training_data, scheduler, epoch, wandb)
        torch.cuda.empty_cache()

        val_loss, val_out, val_target = evaluate(model, val_0_data, criterion, config)
        wandb.log({"val_0_loss": val_loss})
        val_loss, val_out, val_target = evaluate(model, val_1_data, criterion, config)
        wandb.log({"val_1_loss": val_loss})

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            .format(epoch, (time.time() - epoch_start_time),
                                        val_loss))
        print('-' * 89)
        
        

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        plot_interval = 1000

        if epoch % plot_interval == 0:
            train_loss, train_out, train_target = evaluate(model, training_data, criterion, config)
            plotting(val_target, val_out, local, log_name, epoch, val_loss, train_loss, train_out, train_target)
            
    ## End Training

    print('-' * 89)
    print('| End of training | time: {:5.2f}s |'.format((time.time() - overall_start_time)))
    print('-' * 89)
    print("Best validation loss {:.4f}".format(best_val_loss))

    if local:
        path = '../Models/'
    else:
        path = "/mnt/xprun/out/"

    date = datetime.datetime.now().strftime("%Y%m%d%H")
    torch.save(best_model.state_dict(), path + config.xp_name +'.pth')
    # save config dict with pickle
    with open(path + config.xp_name + '.pkl', 'wb') as f:
        pickle.dump(config, f)
    

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
