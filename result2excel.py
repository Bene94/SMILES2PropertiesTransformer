import numpy as np
import pandas as pd

from data_processing.data_processing_cosmo import *

## function to load the output of the modle and put it in a human readable excel file

name = '211118-063721'
save_path = '/home/bene/NNGamma/temp/'
vocab_path = "../vocab/"

train_out = np.load(save_path + 'train_out.npy')
train_target = np.load(save_path + 'train_target.npy')
train_smile = np.load(save_path + 'train_smile.npy')
train_xT = np.load(save_path + 'train_xT.npy')

val_0_out = np.load(save_path + 'val_0_out.npy')
val_0_target = np.load(save_path + 'val_0_target.npy')
val_0_smile = np.load(save_path + 'val_0_smile.npy')
val_0_xT = np.load(save_path + 'val_0_xT.npy')

val_1_out = np.load(save_path + 'val_1_out.npy')
val_1_target = np.load(save_path + 'val_1_target.npy')
val_1_smile = np.load(save_path + 'val_1_smile.npy')
val_1_xT = np.load(save_path + 'val_1_xT.npy')

val_2_out = np.load(save_path + 'val_2_out.npy')
val_2_target = np.load(save_path + 'val_2_target.npy')
val_2_smile = np.load(save_path + 'val_2_smile.npy')
val_2_xT = np.load(save_path + 'val_2_xT.npy')

# turn np arrays into pandas dataframes for train and val

train_df = pd.DataFrame(data={ 'smile':train_smile.tolist(),'x':train_xT[:,0], 'T':train_xT[:,1], 'out':train_out, 'target':train_target })
val_0_df = pd.DataFrame(data={ 'smile':val_0_smile.tolist(),'x':val_0_xT[:,0], 'T':val_0_xT[:,1], 'out':val_0_out, 'target':val_0_target })
val_1_df = pd.DataFrame(data={ 'smile':val_1_smile.tolist(),'x':val_1_xT[:,0], 'T':val_1_xT[:,1], 'out':val_1_out, 'target':val_1_target })
val_2_df = pd.DataFrame(data={ 'smile':val_2_smile.tolist(),'x':val_2_xT[:,0], 'T':val_2_xT[:,1], 'out':val_2_out, 'target':val_2_target })

# append the dataframes to one dataframe
df = pd.concat([train_df, val_0_df, val_1_df, val_2_df])
df.reset_index(drop=True, inplace=True)

print("Data Loading")
vocab_dict = load_vocab(vocab_path,'vocab_dict_aug')

df = revert_vocab(df, vocab_dict)
df.to_excel(save_path + name + '.xlsx', index=False)
# save high error data to excel
#df = df[ np.abs(df['out']-df['target']) > 0.5]
#df.to_excel(save_path + name + '_high_error.xlsx', index=False)


