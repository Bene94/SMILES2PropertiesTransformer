
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
    path = '/home/bene/NNGamma/Models/'
    name = '211107-220344'
    save_path = '/home/bene/NNGamma/temp/'
    model, config = load_model(path,name)

    calc = True
    save = False

    if calc:
        #model to devide
        print(config.data_path)
        model = model.to('cuda')

        criterion = nn.MSELoss()

        #data_path = os.path.join('/home/bene/NNGamma/data/' + config.data_path + '/')
        data_path = os.path.join('/home/bene/NNGamma/data/exp/')
        #data_path = os.path.join('/home/bene/NNGamma/data/data_no_tail/')

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

        train_loss, train_out, train_target = evaluate(model, training_data, criterion, config)

        print('-' * 89)
        print('Calculating Validation...')
        print('-' * 89)

        val_0_loss, val_0_out, val_0_target = evaluate(model, val_0_data, criterion, config)
        val_1_loss, val_1_out, val_1_target = evaluate(model, val_1_data, criterion, config)
        val_2_loss, val_2_out, val_2_target = evaluate(model, val_2_data, criterion, config)

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


        if save:
            # save the results to a file
            np.save(save_path + 'train_out.npy', train_out)
            np.save(save_path + 'train_target.npy', train_target)


            np.save(save_path + 'val_0_out.npy', val_0_out)
            np.save(save_path + 'val_0_target.npy', val_0_target)
            np.save(save_path + 'val_1_out.npy', val_1_out)
            np.save(save_path + 'val_1_target.npy', val_1_target)
            np.save(save_path + 'val_2_out.npy', val_2_out)
            np.save(save_path + 'val_2_target.npy', val_2_target)

    else:

        train_out = np.load(save_path + 'train_out.npy')
        train_target = np.load(save_path + 'train_target.npy')
        val_0_out = np.load(save_path + 'val_0_out.npy')
        val_0_target = np.load(save_path + 'val_0_target.npy')
        val_1_out = np.load(save_path + 'val_1_out.npy')
        val_1_target = np.load(save_path + 'val_1_target.npy')
        val_2_out = np.load(save_path + 'val_2_out.npy')
        val_2_target = np.load(save_path + 'val_2_target.npy')


    if False:
        make_MSE_x(train_out, train_target, name = "train", save = True)
        make_MSE_x(val_0_out, val_0_target, name = "val_0", save = True)
        make_MSE_x(val_1_out, val_1_target, name = "val_1", save = True)



        make_heatmap(train_out, train_target, name = "train", save = True)
        make_heatmap(val_0_out, val_0_target, name = "val_0", save = True)
        make_heatmap(val_1_out, val_1_target, name = "val_1", save = True)
        make_heatmap(val_2_out, val_2_target, name = "val_2", save = True)

        make_historgam_delta(train_out, train_target, name = "train", save = True)
        make_historgam_delta(val_0_out, val_0_target, name = "val_0", save = True)
        make_historgam_delta(val_1_out, val_1_target, name = "val_1", save = True)
        make_historgam_delta(val_2_out, val_2_target, name = "val_2", save = True)

    if len(train_out) < 10000:
        print('-' * 89)
        print('Make Scatter...')
        print('-' * 89)
        # concatenate the resutls
        train_out = np.concatenate((train_out, val_0_out, val_1_out), axis=0)
        train_target = np.concatenate((train_target, val_0_target, val_1_target), axis=0)
        make_scatter(train_out, train_target, name = "train", save = True)
