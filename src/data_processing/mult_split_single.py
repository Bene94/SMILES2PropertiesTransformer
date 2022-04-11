import numpy as np

import data_processing as dp

n_splits = 20
file_path = ["elect"]
save_path = "data_elect"
vocab_path = "vocab"
frac = 0.1
ow = True
h2o = False
source = 'single'
pre = ''

for i in range(n_splits):
    print("\n")
    print(str(i) + "th split \n")
    save_path = "data_elect" + "/" + str(i)
    dp.processing(file_path, save_path, vocab_path, frac, i, ow, h2o, source, pre)