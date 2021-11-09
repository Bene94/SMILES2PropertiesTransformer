
from dataclasses import dataclass
import matplotlib as mpl
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from nn_dataloader import *

@dataclass()
class Config:
    data_path: str
    data_path: str
    shift: int
    mode: str = 'reg'
    batch_size: int = 1024
    test = False



## function to compare the data from experimental and simulation data

path_data = os.path.join('/home/bene/NNGamma/data/data/')

config = Config(data_path=path_data, shift=0)

data = gamma_dataset(path_data, '', config)

target = data.train_target
xT = data.xT
smile = data.train_data
x = xT[:,0]
target = torch.squeeze(target)

# plot target over x, x is in range 0,1


fig, ax = plt.subplots()
ax.plot(x, target, '*b', label='target')
ax.set_xlabel('x')
ax.set_ylabel('target')
ax.set_title('Target over x')
ax.legend()
plt.show()

#save in the plot/ folder
plt.savefig('plot/target_over_x.png')



