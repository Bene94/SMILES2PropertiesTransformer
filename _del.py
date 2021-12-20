
## load the pytorch transformer model and the cofiguration file

import pickle
from numpy.random.mtrand import random
import torch
import torch.nn as nn
import os

import wandb
import click

import minGPT 
from nn_dataloader import *
from plot_results import *
from trainer import *
from plot_results import *




@click.command()

@click.option('--name','-n', default='211218-225820', help='Name of the modle')
@click.option('--data','-d', default='data_xt', help='Path to the data if empty use datapath from modle config')

@click.option('--calc','-c', default=True, help='Calculate results and eval')
@click.option('--plot','-p', default=False, help='Plot results')
@click.option('--save','-s', default=True, help='Save results')


def main(name,data,calc,plot,save):
    
    if os.environ.get('XPRUN_NAME') is not None:
        local = False
        path_temp = "/mnt/xprun/temp/"
        path_model = "/mnt/xprun/out/"
        path_wandb = "/mnt/xprun/wandb/"
        data_path = "/mnt/xprun/data/" + data + "/"
        save_path = "/mnt/xprun/out/" + name + "/"
        xp_name = os.environ['XPRUN_NAME']
    else:
        local = True
        path_temp = '../temp/'
        path_model = '../Models/'
        path_wandb = '../wandb/'
        data_path = '../data/' + data + '/'
        save_path = '../out/' + name +  '/'
        xp_name = 'local_test' + str(random())

    model, config = load_model(path_model,name)

    if calc:

        #model to devide
        print(config.data_path)
        model = model.to('cuda')
        criterion = nn.MSELoss()

        print('-' * 89)
        print('Loading Data...')
        print('-' * 89)

        train_dataset = gamma_dataset(data_path, 'train', config)
        val_0_dataset = gamma_dataset(data_path, 'val_0', config)
        val_1_dataset = gamma_dataset(data_path, 'val_1', config)
        val_2_dataset = gamma_dataset(data_path, 'val_2', config)

        training_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        val_0_data = DataLoader(val_0_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        val_1_data = DataLoader(val_1_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        val_2_data = DataLoader(val_2_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

        print('-' * 89)
        print('Calculating Traning...')
        print('-' * 89)

        train_loss, train_out, train_target, train_in = evaluate(model, training_data, criterion, config)

        print('-' * 89)
        print('Calculating Validation...')
        print('-' * 89)

        val_0_loss, val_0_out, val_0_target, val_0_in = evaluate(model, val_0_data, criterion, config)
        val_1_loss, val_1_out, val_1_target, val_1_in = evaluate(model, val_1_data, criterion, config)
        val_2_loss, val_2_out, val_2_target, val_2_in = evaluate(model, val_2_data, criterion, config)

        train_target = train_target.squeeze()
        train_out = train_out.squeeze()

        val_0_target = val_0_target.squeeze()
        val_0_out = val_0_out.squeeze()
        val_1_target = val_1_target.squeeze()
        val_1_out = val_1_out.squeeze()
        val_2_target = val_2_target.squeeze()
        val_2_out = val_2_out.squeeze()

        print("Training loss: ", train_loss)
        print("Validation loss: ", val_0_loss)
        print("Validation loss: ", val_1_loss)
        print("Validation loss: ", val_2_loss)

        # find index of nan in output
        train_nan = np.isnan(train_out)
        val_0_nan = np.isnan(val_0_out)
        val_1_nan = np.isnan(val_1_out)
        val_2_nan = np.isnan(val_2_out)


def load_model(path, name):
    # load config file
    #list all files in path
    files = os.listdir(path)
    # check if the beginning of one file is the same as the name else make error
    for file in files:
        if file.startswith(name):
            config_file = file
    # remove last 4 characters from config file name
    config_file = config_file[:-4]
    config = pickle.load(open(path + config_file + '.pkl', 'rb'))

    #config = convert_config(config)

    # load model
    model = minGPT.GPT(config)
    model.load_state_dict(torch.load(path + config_file + '.pth'))
    model.eval()
    return model, config

def convert_config(config):
    wandb.init(config=config)
    config = wandb.config
    wandb.finish()
    return config


if __name__ == '__main__':
    main()
