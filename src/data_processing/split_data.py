import numpy as np
import pandas as pd
import os
import progressbar as pb

import data_processing as dp

# Function to take data with two outputs and convert it to a single output
folder_name = ["brouwer"]
vocab_path = "vocab"
save_path = "brouwer_split/cosmo_data_0"

file_path, file_out, vocab_path, alias_path  =  dp.get_paths(save_path, vocab_path) 
df_join, comp_list, index_list  = dp.load_exp_data(file_path, folder_name)
df_join['y1'] = 0
df_join = df_join[['SMILES0','SMILES1','y0','y1','x','T','i']]
df_2 = df_join.copy()
df_2.rename(columns={'SMILES0':'SMILES1','SMILES1':'SMILES0','y0':'y1','y1':'y0'}, inplace=True)
# reorder columns
df_2['x'] = 1


df_new = df_join.append(df_2, ignore_index=True)

df_new.to_csv(os.path.join(file_path,save_path), index=False)