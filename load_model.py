
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

def load_model(path, name):
    # load config file
    config_path = os.path.join(path + name + '.pkl')
    with open(config_path, 'rb') as f:
        config = pickle.load(f)

    config = convert_config(config)

    # load model
    model = minGPT.GPT(config)
    model.load_state_dict(torch.load(path + name + '.pth'))
    model.eval()
    return model, config

def convert_config(config):
    wandb.init(config=config)
    config = wandb.config
    return config

if __name__ == '__main__':
    path = '/home/bene/NNGamma/Models/'
    name = '2021082507_minGPT_512_2_8_1e-01_1e-01_1e-04_1024_10'
    model, config = load_model(path,name)
    print("done")

    #model to devide
    model = model.to('cuda')

    if config.criterion == 'MSELoss()':
        criterion = nn.MSELoss()

    data_path = os.path.join('/home/bene/NNGamma/' + config.data_path + '/')

    train_dataset = gamma_dataset(data_path, 'train')
    val_dataset = gamma_dataset(data_path, 'val')

    training_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    val_data = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    val_loss, val_out, val_target = evaluate(model, val_data, criterion, config)
    train_loss, train_out, train_target = evaluate(model, training_data, criterion, config)

    print("Validation loss: ", val_loss)
    print("Training loss: ", train_loss)

    print("Expected validation loss: ", config.val_loss)


