
from dataclasses import dataclass
import matplotlib.pyplot as plt
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import progressbar as pb

from transprop.nn_dataloader import *

@dataclass()
class Config:
    data_path: str
    data_path: str
    shift: int
    mode: str = 'reg'
    batch_size: int = 1024
    test = False

## function to compare the data from experimental and simulation data

len_train = []
len_val0 = []
len_val1 = []
len_val2 = []

bar = pb.ProgressBar(maxval=1000, widgets=[pb.Bar('=', '[', ']'), ' ', pb.Percentage(), ' ', pb.ETA()])
bar.start()
for i in range(0,1000):
    bar.update(i)
    path_data = os.path.join('/home/bene/NNGamma/data/data_exp_noH2O_1000_V2/' + str(i) + '/')

    config = Config(data_path=path_data, shift=0)

    train = gamma_dataset(path_data, 'train', config)
    val0 = gamma_dataset(path_data, 'val_0', config)
    val1 = gamma_dataset(path_data, 'val_1', config)
    val2 = gamma_dataset(path_data, 'val_2', config)

    len_train.append(len(train))
    len_val0.append(len(val0))
    len_val1.append(len(val1))
    len_val2.append(len(val2))
bar.finish()
    
# print average,  min and max length of the datasets in a nice table
print('\n')
print('Average length of the datasets:')
print('Train:', np.mean(len_train))
print('Val0:', np.mean(len_val0))
print('Val1:', np.mean(len_val1))
print('Val2:', np.mean(len_val2))
print('\n')
print('Min length of the datasets:')
print('Train:', np.min(len_train))
print('Val0:', np.min(len_val0))
print('Val1:', np.min(len_val1))
print('Val2:', np.min(len_val2))
print('\n')
print('Max length of the datasets:')
print('Train:', np.max(len_train))
print('Val0:', np.max(len_val0))
print('Val1:', np.max(len_val1))
print('Val2:', np.max(len_val2))
print('\n')
