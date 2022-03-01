import data_processing as dc
import numpy as np
import os
from numpy.random import default_rng
## function to create multiple splits of the data
file_path = ["brouwer_exp_c"]
save_path = "data_exp"
vocab_path = "vocab"
ow = True

num_splits = 1000

data_path = '../data/'
#data_path = '/mnt/xprun/data/'

comp_list, n_unique = dc.get_comp_list(file_path, vocab_path)
index_list = np.arange(0,n_unique)
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
    dc.processing_n_out(file_path, save_path_temp, vocab_path, ow, comp_list, index)