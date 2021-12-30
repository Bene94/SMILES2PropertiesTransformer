import numpy as np
import pandas as pd

from data_processing.data_processing import *

## function to load the output of the modle and put it in a human readable excel file

name = '211126-160520'
name = '211220-192228' # modle with leave n out

save_path = '/home/bene/NNGamma/out/' + name + '/'
data_path = '/home/bene/NNGamma/data/exp_t/'
vocab_path = "../vocab/"

train_out = np.load(save_path + 'train_out.npy')
train_target = np.load(save_path + 'train_target.npy')
train_smile = np.load(save_path + 'train_smile.npy')
train_xT = np.load(save_path + 'train_xT.npy')
train_x = np.load(save_path + 'train_x.npy')

val_0_out = np.load(save_path + 'val_0_out.npy')
val_0_target = np.load(save_path + 'val_0_target.npy')
val_0_smile = np.load(save_path + 'val_0_smile.npy')
val_0_xT = np.load(save_path + 'val_0_xT.npy')
val_0_x = np.load(save_path + 'val_0_x.npy')

val_1_out = np.load(save_path + 'val_1_out.npy')
val_1_target = np.load(save_path + 'val_1_target.npy')
val_1_smile = np.load(save_path + 'val_1_smile.npy')
val_1_xT = np.load(save_path + 'val_1_xT.npy')
val_1_x = np.load(save_path + 'val_1_x.npy')

val_2_out = np.load(save_path + 'val_2_out.npy')
val_2_target = np.load(save_path + 'val_2_target.npy')
val_2_smile = np.load(save_path + 'val_2_smile.npy')
val_2_xT = np.load(save_path + 'val_2_xT.npy')
val_2_x = np.load(save_path + 'val_2_x.npy')

# turn np arrays into pandas dataframes for train and val

train_df = pd.DataFrame(data={ 'solute_index':train_smile[:,0].tolist(),'solvent_index':train_smile[:,1].tolist(),'x':train_xT[:,0], 'T':train_xT[:,1], 'out':train_out, 'target':train_target, 'x_index':train_x})
val_0_df = pd.DataFrame(data={ 'solute_index':val_0_smile[:,0].tolist(),'solvent_index':val_0_smile[:,1].tolist(),'x':val_0_xT[:,0], 'T':val_0_xT[:,1], 'out':val_0_out, 'target':val_0_target, 'x_index':val_0_x})
val_1_df = pd.DataFrame(data={ 'solute_index':val_1_smile[:,0].tolist(),'solvent_index':val_1_smile[:,1].tolist(),'x':val_1_xT[:,0], 'T':val_1_xT[:,1], 'out':val_1_out, 'target':val_1_target, 'x_index':val_1_x})
val_2_df = pd.DataFrame(data={ 'solute_index':val_2_smile[:,0].tolist(),'solvent_index':val_2_smile[:,1].tolist(),'x':val_2_xT[:,0], 'T':val_2_xT[:,1], 'out':val_2_out, 'target':val_2_target, 'x_index':val_2_x})


# append the dataframes to one dataframe
df = pd.concat([train_df, val_0_df, val_1_df, val_2_df])
df.reset_index(drop=True, inplace=True)

print("Data Loading")
comp_list = pd.read_csv(data_path + 'comp_list.csv')

df = revert_vocab(df, comp_list)
df.to_excel(save_path + name + '.xlsx', index=False)
# save high error data to excel
#df = df[ np.abs(df['out']-df['target']) > 0.5]
#df.to_excel(save_path + name + '_high_error.xlsx', index=False)


