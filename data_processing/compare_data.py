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

path_sim = os.path.join('/home/bene/NNGamma/data/' +'data/')
path_exp = os.path.join('/home/bene/NNGamma/data/' +'exp_t/')



config_sim = Config(data_path=path_sim, shift=0)
config_exp = Config(data_path=path_exp, shift=0)

exp_data = gamma_dataset(path_exp, '', config_exp)
exp_dict = dict(zip(exp_data.train_data, exp_data.train_target))

sim_data = gamma_dataset(path_sim, '', config_sim)
sim_dict = dict(zip(sim_data.train_data, sim_data.train_target))

# get thte data that is in both datasets and plot the targets
targets_exp = []
targets_sim = []
for key in exp_dict:
    if key in sim_dict:
        targets_exp.append(exp_dict[key])
        targets_sim.append(sim_dict[key])

# plot the data
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(targets_exp, targets_sim, s=1)
ax.set_xlabel('Experimental data')
ax.set_ylabel('Simulation data')
ax.set_title('Comparison of experimental and simulation data')
plt.show()
# save plot
fig.savefig('compare_data.png')

pass 








