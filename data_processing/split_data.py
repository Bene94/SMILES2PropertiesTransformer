import numpy as np
import pandas as pd
import os
import progressbar as pb

import data_processing as dp

# Function to take data with two outputs and convert it to a single output
folder_name = ["x_cosmo"]
vocab_path = "vocab"
save_path = "x_cosmo_split/cosmo_data_0"

file_path, file_out, vocab_path, alias_path  =  dp.get_paths(save_path, vocab_path) 
df_join, comp_list, solvent_indx, solute_indx  = dp.load_exp_data(file_path, folder_name)
df_2 = df_join.copy()

# rename the column lnGamma_1 to lnGamma
df_join.rename(columns={'lnGamma_1':'lnGamma'}, inplace=True)
# drop column lnGamma_2
df_join.drop(columns=['lnGamma_2'], inplace=True)
df_2.rename(columns={'lnGamma_2':'lnGamma'}, inplace=True)
df_2.rename(columns={'solvent':'temp'}, inplace=True)
df_2.rename(columns={'solute':'solvent'}, inplace=True)
df_2.rename(columns={'temp':'solute'}, inplace=True)
df_2.drop(columns=['lnGamma_1'], inplace=True)
df_2.x = 1 - df_2.x

df_new = df_join.append(df_2, ignore_index=True)

df_new.to_csv(os.path.join(file_path,save_path), index=False)