import data_processing as dc
import numpy as np
import os

## function to create multiple splits of the data


file_path = ["brouwer_exp"]
save_path = "data_exp_n"
vocab_path = "vocab"
ul = np.inf
ll = -np.inf
frac = 0.05
aug = False
max_aug = 10
h2o = True # if True, H2O can be in the validation set

num_splits = 200

#n_list = [10, 50 , 100, 500, 1000, 5000, 10000]

n_list = [3, 5, 20 , 30, 40]

for n in n_list:
    for i in range(0,num_splits):
        print("\n")
        print(str(i) + "th split \n")

        save_path_temp = save_path +"/" + "n_" + str(n) + "/" + str(i)
        # check if the directory exists
        if not os.path.exists('../data/'+save_path_temp):
            os.makedirs('../data/'+save_path_temp)
        dc.processing_n(file_path, save_path_temp, vocab_path, ul, ll, frac, aug,  max_aug, i, True, h2o, n)