import numpy as np
import pandas as pd

from data_processing.data_processing_cosmo import *

## function to load the output of the modle and put it in a human readable excel file

name = '211118-063721'
save_path = '/home/bene/NNGamma/temp/'
vocab_path = "../vocab/"

train_out = np.load(save_path + 'train_out.npy')
train_target = np.load(save_path + 'train_target.npy')
val_0_out = np.load(save_path + 'val_0_out.npy')
val_0_target = np.load(save_path + 'val_0_target.npy')
val_1_out = np.load(save_path + 'val_1_out.npy')
val_1_target = np.load(save_path + 'val_1_target.npy')
val_2_out = np.load(save_path + 'val_2_out.npy')
val_2_target = np.load(save_path + 'val_2_target.npy')

# turn np arrays into pandas dataframes
train_out_df = pd.DataFrame(train_out)
train_target_df = pd.DataFrame(train_target)


print("Data Loading")
vocab_dict = load_vocab(vocab_path,'vocab_dict_aug')


