
## load the pytorch transformer model and the cofiguration file

import pickle
import torch
import os

import wandb

import minGPT as minGPT 
from nn_dataloader import *
from trainer import *

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