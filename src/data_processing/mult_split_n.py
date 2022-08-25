import data_processing as dc
import numpy as np
import os

## function to create multiple splits of the data


file_path = ["brouwer"]
save_path = "data_exp_n"
vocab_path = "vocab"
ul = np.inf
ll = -np.inf
frac = 0.05
aug = False
max_aug = 10
h2o = True # if True, H2O can be in the validation set

num_splits = 200

n_list = [2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200, 300, 400, 500, 600, 700, 800, 1000, 1500, 2000, 3000, 4000, 5000]

comp_list, __, ___ = dc.get_comp_list(file_path, vocab_path)

for n in n_list:
    for i in range(0,num_splits):
        print("\n")
        print(str(i) + "th split \n")

        save_path_temp = save_path +"/" + "n_" + str(n) + "/" + str(i)
        # check if the directory exists
        if not os.path.exists('../data/'+save_path_temp):
            os.makedirs('../data/'+save_path_temp)
        dc.processing_n_in(file_path, save_path_temp, vocab_path, i, True, n, comp_list)