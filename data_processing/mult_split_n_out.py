import data_processing as dc
import numpy as np
import os
from numpy.random import default_rng
## function to create multiple splits of the data
file_path = ["brouwer_exp"]
save_path = "data_exp_edgeH2O"
vocab_path = "vocab"
ow = True
exclude_H2O = True
edge_H2O = False

num_splits = 200

data_path = '../data/'
#data_path = '/mnt/xprun/data/'

comp_list, systems = dc.get_comp_list(file_path, vocab_path)
n_unique = len(systems)
index_list = np.arange(0,n_unique)
# find systems with water as solvent
if exclude_H2O:
    h2o_index = systems[systems.solvent == 'O'].index
    index_list = np.setdiff1d(index_list, h2o_index)
elif edge_H2O:
    h2o_index = systems[systems.solvent == 'O'].index
    solutes = systems.solute[h2o_index]
    solutes = solutes.drop_duplicates()
    index_list = systems[systems.solute.isin(solutes)].index
    index_list = np.setdiff1d(index_list, h2o_index)


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
    dc.processing_n_out(file_path, save_path_temp, vocab_path, ow, comp_list, systems,index)