import data_processing as dc
import numpy as np
import os
from numpy.random import default_rng
## function to create multiple splits of the data

file_path = ["brouwer"]
vocab_path = "vocab"
mode = 'brouwer'
ow = True

exclude_H2O = True
only_H2O = False

data_path = '../data/'

comp_list, systems, __ = dc.get_comp_list(file_path, vocab_path)
n_unique = len(systems)
index_list = np.arange(0,n_unique)


# find systems with water as solvent
if exclude_H2O:
    h2o_index = systems[(systems.SMILES0 == 'O') | (systems.SMILES1 == 'O')].index
    index_list = np.setdiff1d(index_list, h2o_index)
    num_splits = 1000
    save_path = "data_exp_noH2O_" + str(num_splits)
    #save_path = "data_exp_sund_" + str(num_splits)
elif only_H2O:
    index_list = systems[systems.solvent == 'O'].index
    index_list = np.array(index_list)
    save_path = "data_exp_onlyH2O"
    num_splits = 20

# shuffle the index list
rng = default_rng(0)
rng.shuffle(index_list)
# break the index list into chunks of size num_splits
index_list_chunks = np.array_split(index_list, num_splits)

for i, index in enumerate(index_list_chunks):
    print("\n")
    print(str(i) + "th split \n")
    save_path_temp = save_path + "/" +str(i)
    # check if the directory exists
    if not os.path.exists(data_path+save_path_temp):
        os.makedirs(data_path+save_path_temp)
    if mode == 'sund':
        dc.processing_n_out_sund(file_path, save_path_temp, vocab_path, ow, comp_list, systems, index)
    else:
        dc.processing_n_out(file_path, save_path_temp, vocab_path, ow, comp_list, systems, index)