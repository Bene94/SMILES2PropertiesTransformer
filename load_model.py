
## load the pytorch transformer model and the cofiguration file

import pickle
import torch
import torch.nn as nn
import os


import wandb

import minGPT 
from nn_dataloader import *
from plot_results import *
from trainer import *
from plot_results import *

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

    config = convert_config(config)
    config.mode = 'reg'

    # load model
    model = minGPT.GPT(config)
    model.load_state_dict(torch.load(path + config_file + '.pth'))
    model.eval()
    return model, config

def convert_config(config):
    wandb.init(config=config)
    config = wandb.config
    return config

if __name__ == '__main__':
    path = '/home/bene/NNGamma/Models/'
    name = '2021082803_minGPT'
    model, config = load_model(path,name)

    calc = False

    if calc:
        #model to devide
        model = model.to('cuda')

        if config.criterion == 'MSELoss()':
            criterion = nn.MSELoss()

        data_path = os.path.join('/home/bene/NNGamma/' + config.data_path + '/')

        print('-' * 89)
        print('Loading Data...')
        print('-' * 89)

        train_dataset = gamma_dataset(data_path, 'train', config)
        val_dataset = gamma_dataset(data_path, 'val', config)


        x = 100

        if True:
            train_dataset.train_data = train_dataset.train_data[:]
            train_dataset.train_target = train_dataset.train_target[:]

            val_dataset.train_data = val_dataset.train_data[:]
            val_dataset.train_target = val_dataset.train_target[:]

        training_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        val_data = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

        print('-' * 89)
        print('Calculating Validation...')
        print('-' * 89)

        val_loss, val_out, val_target = evaluate(model, val_data, criterion, config)

        print('-' * 89)
        print('Calculating Traning...')
        print('-' * 89)

        train_loss, train_out, train_target = evaluate(model, training_data, criterion, config)

        val_target = val_target.squeeze()
        val_out = val_out.squeeze()

        train_target = train_target.squeeze()
        train_out = train_out.squeeze()

        print("Validation loss: ", val_loss)
        print("Training loss: ", train_loss)

        # save the results to a file
        np.save('val_out.npy', val_out)
        np.save('val_target.npy', val_target)
        np.save('train_out.npy', train_out)
        np.save('train_target.npy', train_target)
    else:
        val_out = np.load('val_out.npy')
        val_target = np.load('val_target.npy')
        train_out = np.load('train_out.npy')
        train_target = np.load('train_target.npy')


    # %% 


    make_MSE_x(val_out, val_target, name = "val", save = True)
    make_MSE_x(train_out, train_target, name = "train", save = True)



    print('-' * 89)
    print('Make Scatter...')
    print('-' * 89)

    make_heatmap(val_out, val_target, name = "val", save = True)
    make_heatmap(train_out, train_target, name = "train", save = True)
