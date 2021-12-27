import data_processing as dc
import numpy as np
import os

## function to create multiple splits of the data


file_path = ["brouwer_exp"]
save_path = "data_exp"
vocab_path = "vocab"
ul = np.inf
ll = -np.inf
frac = 0.03
aug = False
max_aug = 10
h2o = True # if True, H2O can be in the validation set

num_splits = 500

data_path = '../data/'
#data_path = '/mnt/xprun/data/'

for i in range(0,num_splits):
    print("\n")
    print(str(i) + "th split \n")

    save_path_temp = save_path + "/" +str(i)
    # check if the directory exists
    if not os.path.exists(data_path+save_path_temp):
        os.makedirs(data_path+save_path_temp)
    dc.processing(file_path, save_path_temp, vocab_path, ul, ll, frac, aug,  max_aug, i, ow=True, h2o=h2o, source='EXP')