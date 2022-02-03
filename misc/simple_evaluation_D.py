# %% 
# This is a simple script to evaluate the modle for custom data pairs
import os
import sys

# add current path to sys.path
sys.path.append(os.getcwd())
# %% 
# This is a simple script to evaluate the modle for custom data pairs


# %% Imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from transprop.trainer import *
from transprop.nn_dataloader import *
from transprop.load_model import *
from simple_evaluation_utils import *

# %% Parameters
model_path = '/home/bene/NNGamma/Models/'
model_name = 'local_test_fine'
model_name =  '211123-110140'
model_name = 'f_t_220127-180116_220131-015826'
device = 'cuda'
device = 'cpu'

solvent = "C1CCCCC1"
solvent = "CC(C)=O"
solvent = "CC(=O)C"
solute = "c1ccccc1C"
solute = "Cc1ccccc1"

T = 298.15

x = np.linspace(0, 1, num=20)
T = np.linspace(T, T, 1)

data_loader_solute = smile2input(solute, solvent, x ,T)
data_loader_solvent = smile2input(solvent, solute, 1-x ,T)
# %% Load model

model, config = load_model(model_path,model_name)
config.device = device
model = model.to(config.device)
criterion = nn.MSELoss()
# time evaluation
start = time.time()
__,gamma_solute, __, __ = evaluate(model, data_loader_solute, criterion, config)
__,gamma_solvent, __, __ = evaluate(model, data_loader_solvent, criterion, config)
mean = (gamma_solute + np.flip(gamma_solvent)) / 2
end = time.time()
print("Evaluation time: ", end - start)

# %%

messure = np.array([2.78,2.71,2.60,2.54,2.62,2.75,2.86,3.11,3.46])
messure_err = np.array([1.58,1.05,0.79,0.09,0.68,0.80,0.75,0.90,0.93]) * 0.1
x_mesure =  np.linspace(0.1, 0.9, num=9)

MD = np.array([2.74,	2.9,	2.76,	2.59,	2.6,	2.69,	2.88,	3,	3.11,	3.18,	3.45])
MD_x = np.array([0.05,	0.1,	0.2,	0.3,	0.4,	0.5,	0.6,	0.7,	0.8,	0.9,	0.95])
# plot train out over x 
fig, ax = plt.subplots()
# use dashed lines
ax.fill_between(x, gamma_solute, np.flip(gamma_solvent), alpha=0.5, color='b')
ax.plot(x, mean, 'b', label='Solute')

ax.errorbar(x_mesure, messure, yerr=messure_err, fmt='o')
ax.errorbar(MD_x, MD, yerr=0.0, fmt='x')

ax.set_title(str(T[0]) + " K")
ax.set_xlabel('x ' + solvent )
ax.set_ylabel('D')
ax.set_xlim([0,1])
ax.legend([solute, solvent, 'mean', 'carsten', 'MD'])
plt.show()

plt.savefig('plot/simple_.png')