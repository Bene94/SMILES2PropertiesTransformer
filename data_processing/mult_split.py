import data_processing_cosmo as dc
import numpy as np

## function to create multiple splits of the data


file_path = "brouwer_exp"
save_path = "data_exp"
vocab_path = "vocab"
ul = np.inf
ll = -np.inf
frac = 0.05
aug = False

num_splits = 200

for i in range(num_splits):
    print("\n")
    print(str(i) + "th split \n")

    save_path_temp = save_path + "/" +str(i)  
    dc.processing(file_path, save_path_temp, vocab_path, ul, ll, frac, aug, i, ow=True)